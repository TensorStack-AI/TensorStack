using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text.Json.Serialization;
using TensorStack.WPF;
using TensorStack.Example.Common;
using TensorStack.Common;
using TensorStack.Providers;

namespace TensorStack.Example
{
    public class Settings : IUIConfiguration
    {
        [JsonIgnore]
        public Device DefaultDevice { get; set; }
        public int ReadBuffer { get; set; } = 32;
        public int WriteBuffer { get; set; } = 32;
        public string VideoCodec { get; set; } = "mp4v";
        public string DirectoryTemp { get; set; }
        public IReadOnlyList<Device> Devices { get; set; }
        public ObservableCollection<ExtractorModel> ExtractorModels { get; set; }

        public void Initialize()
        {
            Directory.CreateDirectory(DirectoryTemp);

            Provider.Initialize();
            Devices = Provider.GetDevices();
            DefaultDevice = Provider.GetDevice();
        }
    }
}
