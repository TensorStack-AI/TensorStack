using System.Collections.Generic;


namespace Amuse.Common
{
    public sealed record PipelineLoadOptions
    {
        public string ModelPath { get; set; }
        public string Template { get; set; }
        public string ModelType { get; set; }
        public string Variant { get; set; }
        public string Pipeline { get; set; }
        public ProcessType ProcessType { get; set; }
        public string Device { get; set; }
        public int DeviceId { get; set; }
        public int DeviceBusId { get; set; }
        public DataType DataType { get; set; }
        public QuantizationType QuantType { get; set; }
        public bool IsOptimizeDeviceEnabled { get; set; } = false;
        public bool IsOptimizeChannelsEnabled { get; set; } = false;
        public bool IsDeviceQuantizationEnabled { get; set; } = false;
        public List<LoraConfig> LoraAdapters { get; set; }
        public ControlNetConfig ControlNet { get; set; }
        public MemoryModeType MemoryMode { get; set; }
        public CheckpointConfig CheckpointConfig { get; set; }
    }
}
