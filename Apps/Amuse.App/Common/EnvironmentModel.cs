using Amuse.Common;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;
using TensorStack.Common;
using TensorStack.WPF;

namespace Amuse.App.Common
{
    public sealed class EnvironmentModel : BaseModel
    {
        private string _name;
        private EnvironmentType _type;
        private VendorType _vendor;
        private int _device;
        private PipelineType? _pipeline;
        private bool _isDefault;
        private string _environment;
        private string _pythonVersion;

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int Id { get; set; }
        public int Version { get; set; }
        public EnvironmentMode Status { get; set; }

        public string Name
        {
            get { return _name; }
            set { SetProperty(ref _name, value); }
        }

        public EnvironmentType Type
        {
            get { return _type; }
            set { SetProperty(ref _type, value); }
        }

        public VendorType Vendor
        {
            get { return _vendor; }
            set { SetProperty(ref _vendor, value); }
        }

        public int Device
        {
            get { return _device; }
            set { SetProperty(ref _device, value); }
        }

        public PipelineType? Pipeline
        {
            get { return _pipeline; }
            set { SetProperty(ref _pipeline, value); }
        }

        public bool IsDefault
        {
            get { return _isDefault; }
            set { SetProperty(ref _isDefault, value); }
        }

        public string Environment
        {
            get { return _environment; }
            set { SetProperty(ref _environment, value); }
        }

        public string PythonVersion
        {
            get { return _pythonVersion; }
            set { SetProperty(ref _pythonVersion, value); }
        }

        public string[] Requirements { get; set; }
        public Dictionary<string, string> Variables { get; set; }


        public EnvironmentModel DeepClone(int id)
        {
            return new EnvironmentModel
            {
                Id = id,
                Name = Name,
                IsDefault = IsDefault,
                Environment = Environment,
                PythonVersion = PythonVersion,
                Vendor = Vendor,
                Variables = Variables?.ToDictionary() ?? new Dictionary<string, string>(),
                Requirements = Requirements.ToArray(),
                Pipeline = Pipeline,
                Device = Device,
                Type = Type
            };
        }
    }
}
