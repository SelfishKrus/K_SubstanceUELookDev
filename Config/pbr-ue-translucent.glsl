/*************************************************************************
* ADOBE CONFIDENTIAL
* ___________________
* Copyright 2014 Adobe
* All Rights Reserved.
* NOTICE:  All information contained herein is, and remains
* the property of Adobe and its suppliers, if any. The intellectual
* and technical concepts contained herein are proprietary to Adobe
* and its suppliers and are protected by all applicable intellectual
* property laws, including trade secret and copyright laws.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Adobe.
*************************************************************************/

//- Substance 3D Painter Metal/Rough and opacity PBR shader
//- ================================================
//-
//- Import from libraries.
import lib-pbr.glsl
import lib-bent-normal.glsl
import lib-emissive.glsl
import lib-pom.glsl
import lib-utils.glsl

// Link Metal/Roughness MDL for Iray
//: metadata {
//:   "mdl":"mdl::alg::materials::physically_metallic_roughness::physically_metallic_roughness"
//: }

//- Show back faces as there may be holes in front faces.
//: state cull_face off

//- Enable alpha blending
//: state blend over

//- Channels needed for metal/rough workflow are bound here.
//: param auto channel_basecolor
uniform SamplerSparse basecolor_tex;
//: param auto channel_roughness
uniform SamplerSparse roughness_tex;
//: param auto channel_metallic
uniform SamplerSparse metallic_tex;
//: param auto channel_specularlevel
uniform SamplerSparse specularlevel_tex;
//: param auto channel_opacity
uniform SamplerSparse opacity_tex;
//: param custom { "default": "texture_name", "label": "Texture", "usage": "texture" } 
uniform sampler2D u_sampler_BrdfLUT;

//: param custom { "default": 0.18, "label": "Env diffuse", "widget": "color" } 
uniform vec3 u_envDiffuse; 

//: param custom { "default": true, "label": "Enable Sunlight" }
uniform bool u_bool_sunlight;
//: param custom { "default": 1, "label": "Sunlight Color", "widget": "color" } 
uniform vec3 u_color_sunlight_color; 
//: param custom { "default": 3.14, "label": "Sunlight Intensity" } 
uniform float u_spin_sunlight_intensity; 
//: param custom { "default": 0, "label": "Sunlight Rotation", "min": 0.0, "max": 360 } 
uniform float u_slider_sunlight_rotation; 

// *************************************************************************
// Custom shader lib 
#define PI 3.14159265359

float saturate(float val)
{
  return clamp(val, 0.0, 1.0);
}

vec3 saturate(vec3 val)
{
  return clamp(val, 0.0, 1.0);
}

vec2 saturate(vec2 val)
{
  return clamp(val, 0.0, 1.0);
}

float wrap(float val, float wrap)
{
  return (val + wrap) / (1.0 + wrap);
}

// Unreal BRDF.ush
float d_ggx_ue( float a2, float NoH )
{
	float d = ( NoH * a2 - NoH ) * NoH + 1;	// 2 mad
	return a2 / ( PI*d*d );					// 4 mul, 1 rcp
}

vec3 f_schlick(vec3 SpecularColor, float VoH) {
    float Fc = pow(1.0 - VoH, 5.0); // 1 sub, 3 mul
    // Anything less than 2% is physically impossible and is instead considered to be shadowing
    return clamp(50.0 * SpecularColor.g, 0.0, 1.0) * Fc + (1.0 - Fc) * SpecularColor;
}

float v_schlick( float a2, float NoV, float NoL )
{
	float k = sqrt(a2) * 0.5;
	float Vis_SchlickV = NoV * (1 - k) + k;
	float Vis_SchlickL = NoL * (1 - k) + k;
	return 0.25 / ( Vis_SchlickV * Vis_SchlickL );
}

vec3 fresnelSchlickWithRoughness(vec3 F0, float roughness, float cosTheta) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 envBRDF(vec3 SpecularColor, float Roughness, float NoV) {
    // Importance sampled preintegrated G * F
    vec2 uv = clamp(vec2(NoV, Roughness), 0.001, 0.999);
    vec2 AB = textureLod(u_sampler_BrdfLUT, uv, 0.0).rg;

    // Anything less than 2% is physically impossible and is instead considered to be shadowing 
    vec3 GF = SpecularColor * AB.x + clamp(50.0 * SpecularColor.g, 0.0, 1.0) * AB.y;
    return GF;
}

#define REFLECTION_CAPTURE_ROUGHEST_MIP 1.0
#define REFLECTION_CAPTURE_ROUGHNESS_MIP_SCALE 1.2
#define MIP_OFFSET 1.33 

float ComputeReflectionCaptureMipFromRoughness(float Roughness, float CubemapMaxMip)
{
    // Heuristic that maps roughness to mip level
    // This is done in a way such that a certain mip level will always have the same roughness, regardless of how many mips are in the texture
    // Using more mips in the cubemap just allows sharper reflections to be supported
    float LevelFrom1x1 = REFLECTION_CAPTURE_ROUGHEST_MIP - REFLECTION_CAPTURE_ROUGHNESS_MIP_SCALE * log2(max(Roughness, 0.001));
    return CubemapMaxMip - 1.0 - LevelFrom1x1;
}

// *************************************************************************

//- Shader entry point.
void shade(V2F inputs)
{
  // Fetch material parameters, and conversion to the specular/roughness model
  float roughness = getRoughness(roughness_tex, inputs.sparse_coord);
  roughness = clamp(roughness, 0.04, 1.0);
  vec3 baseColor = getBaseColor(basecolor_tex, inputs.sparse_coord);
  float metallic = getMetallic(metallic_tex, inputs.sparse_coord);
  metallic = clamp(metallic, 0.04, 1.0);

  vec3 diffColor = generateDiffuseColor(baseColor, metallic);
  vec3 specColor = mix(vec3(0.04), baseColor, metallic);

  // Get detail (ambient occlusion) and global (shadow) occlusion factors
  // separately in order to blend the bent normals properly
  float shadowFactor = getShadowFactor();
  float occlusion = getAO(inputs.sparse_coord, true, use_bent_normal);
  float specOcclusion = specularOcclusionCorrection(
    use_bent_normal ? shadowFactor : occlusion * shadowFactor,
    metallic,
    roughness);

  LocalVectors vectors = computeLocalFrame(inputs);
  computeBentNormal(vectors,inputs);

  // Feed parameters for a physically based BRDF integration
  alphaOutput(getOpacity(opacity_tex, inputs.sparse_coord));
  // diffuseShadingOutput(occlusion * shadowFactor * pbrComputeDiffuse(getDiffuseBentNormal(vectors), diffColor));
  // specularShadingOutput(specOcclusion * pbrComputeSpecular_noHorizonFading(vectors, specColor, roughness));

  vec3 diffuse = vec3(0.0);
  vec3 specular = vec3(0.0);

  // ue ibl specular 
  // pre
  float sunRot = radians(u_slider_sunlight_rotation);
  mat3 rotationMatrix = mat3(
      cos(sunRot), 0.0, sin(sunRot),
      0.0, 1.0, 0.0,
      -sin(sunRot), 0.0, cos(sunRot)
  );
  vec3 sunDirWS = rotationMatrix * normalize(vec3(0.0, 1.0, 1.0));
  vec3 halfDir = normalize(sunDirWS + vectors.eye);
  vec3 lightColor = u_color_sunlight_color * (u_spin_sunlight_intensity / 3.14);

  float ndv = saturate(dot(vectors.eye, normalize(vectors.normal)));
  float ndlUnclamped = dot(vectors.normal, sunDirWS);
  float ndl = saturate(ndlUnclamped);
  float ndlWrap = wrap(ndl, 1.0);
  float ndh = saturate(dot(vectors.normal, halfDir));
  float vdh = saturate(dot(vectors.eye, halfDir));

  float a = roughness * roughness;
  float a2 = clamp(a * a, 0.001, 1.0);

  // directional diffuse
  if (u_bool_sunlight)
  {
    vec3 directionalDiffuse = diffColor * ndlWrap * lightColor;
    diffuse += directionalDiffuse;
  }

  // directional specular
  if (u_bool_sunlight)
  {
    float d = d_ggx_ue(a2, ndh);
    vec3 f = f_schlick(specColor, vdh);
    float v = v_schlick(a2, ndv, ndl);
    vec3 directionalSpecular = d * f * v * ndl * lightColor;
    specular += directionalSpecular;
  }

  // env specular
  vec3 fre = fresnelSchlickWithRoughness(specColor, roughness, ndv);
  vec3 envBRDF = envBRDF(fre, roughness, ndv);

  float mipLevel = ComputeReflectionCaptureMipFromRoughness(roughness, 11.0);
  vec3 normalES =  worldToEnvSpace(vectors.normal);
  vec3 eyeES =  worldToEnvSpace(vectors.eye);
  vec3 reflectDirES = -reflect(eyeES,normalES);
  vec3 radiance = envSample(reflectDirES, mipLevel);

  vec3 envSpecular = envBRDF * radiance;
  specular += envSpecular * specOcclusion;

  // env diffuse
  vec3 envDiffuse = u_envDiffuse * diffColor;
  diffuse += envDiffuse;

  emissiveColorOutput(pbrComputeEmissive(emissive_tex, inputs.sparse_coord));
  diffuseShadingOutput(diffuse);
  specularShadingOutput(specular);
}
