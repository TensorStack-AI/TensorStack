using System.Linq;
using System.Reflection;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    [JsonPolymorphic(TypeDiscriminatorPropertyName = "Scheduler")]
    [JsonDerivedType(typeof(LMSOptions), "LMS")]
    [JsonDerivedType(typeof(EulerOptions), "Euler")]
    [JsonDerivedType(typeof(EulerAncestralOptions), "EulerAncestral")]
    [JsonDerivedType(typeof(DDPMOptions), "DDPM")]
    [JsonDerivedType(typeof(DDIMOptions), "DDIM")]
    [JsonDerivedType(typeof(KDPM2Options), "KDPM2")]
    [JsonDerivedType(typeof(KDPM2AncestralOptions), "KDPM2Ancestral")]
    [JsonDerivedType(typeof(DDPMWuerstchenOptions), "DDPMWuerstchen")]
    [JsonDerivedType(typeof(LCMOptions), "LCM")]
    [JsonDerivedType(typeof(DPMSolverMultistepOptions), "DPMSolverMultistep")]
    [JsonDerivedType(typeof(DPMSolverSinglestepOptions), "DPMSolverSinglestep")]
    [JsonDerivedType(typeof(DPMSolverSDEOptions), "DPMSolverSDE")]
    [JsonDerivedType(typeof(DEISMultistepOptions), "DEISMultistep")]
    [JsonDerivedType(typeof(EDMEulerOptions), "EDMEuler")]
    [JsonDerivedType(typeof(EDMDPMSolverMultistepOptions), "EDMDPMSolverMultistep")]
    [JsonDerivedType(typeof(FlowMatchEulerOptions), "FlowMatchEuler")]
    [JsonDerivedType(typeof(FlowMatchHeunOptions), "FlowMatchHeun")]
    [JsonDerivedType(typeof(FlowMatchLCMOptions), "FlowMatchLCM")]
    [JsonDerivedType(typeof(PNDMOptions), "PNDM")]
    [JsonDerivedType(typeof(HeunOptions), "Heun")]
    [JsonDerivedType(typeof(UniPCMultistepOptions), "UniPCMultistep")]
    [JsonDerivedType(typeof(IPNDMOptions), "IPNDM")]
    [JsonDerivedType(typeof(CogVideoXDDIMOptions), "CogVideoXDDIM")]
    [JsonDerivedType(typeof(CogVideoXDPMOptions), "CogVideoXDPM")]
    [JsonDerivedType(typeof(HeliosOptions), "Helios")]
    [JsonDerivedType(typeof(HeliosDMDOptions), "HeliosDMD")]
    [JsonDerivedType(typeof(TCDOptions), "TCD")]
    [JsonDerivedType(typeof(SCMOptions), "SCM")]
    [JsonDerivedType(typeof(SASolverOptions), "SASolver")]
    [JsonDerivedType(typeof(LTXEulerAncestralRFOptions), "LTXEulerAncestral")]
    public abstract record SchedulerOptions
    {
        [JsonIgnore]
        public abstract SchedulerType Scheduler { get; }

        protected void ShallowCopyProperties(SchedulerOptions other)
        {
            var props = GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);
            foreach (var prop in props.Where(p => p.CanWrite))
            {
                prop.SetValue(this, prop.GetValue(other));
            }
        }
    }
}
