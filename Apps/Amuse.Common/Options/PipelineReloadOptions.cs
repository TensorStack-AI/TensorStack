using System.Collections.Generic;

namespace Amuse.Common
{
    public sealed record PipelineReloadOptions
    {
        public ProcessType ProcessType { get; set; }
        public ControlNetConfig ControlNet { get; set; }
        public List<LoraConfig> LoraAdapters { get; set; }
    }
}
