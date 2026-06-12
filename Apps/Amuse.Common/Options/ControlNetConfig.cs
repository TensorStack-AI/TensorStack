namespace Amuse.Common
{
    public sealed class ControlNetConfig
    {
        public string Path { get; set; }
        public string Name { get; set; }
        public string Weights { get; set; }

        public bool Invert { get; set; }
        public int LayerCount { get; set; }
        public bool DisableProjections { get; set; }
    }
}
