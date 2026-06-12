namespace Amuse.Common
{
    public sealed record CheckpointConfig
    {
        public string Compute { get; set; }
        public string TextEncoder { get; set; }
        public string TextEncoder2 { get; set; }
        public string TextEncoder3 { get; set; }
        public string Unet { get; set; }
        public string Transformer { get; set; }
        public string Transformer2 { get; set; }
        public string Vae { get; set; }
        public string AudioVae { get; set; }
        public string Vocoder { get; set; }
        public string Connectors { get; set; }
        public string LatentUpsampler { get; set; }
        public string LatentUpsamplerTemporal { get; set; }
        public string ConditionEncoder { get; set; }
        public string AudioTokenizer { get; set; }
        public string AudioDetokenizer { get; set; }
    }
}
