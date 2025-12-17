using System.Collections.Generic;
using System.Text.Json.Serialization;
using TensorStack.Common.Tensor;
using TensorStack.Python.Options;

namespace TensorStack.Python
{
    public interface IPythonMessage
    {
        List<Tensor<float>> Tensors { get; set; }
    }

    public class PythonRequestMessage : IPythonMessage
    {
        [JsonIgnore]
        public List<Tensor<float>> Tensors { get; set; }

        public PythonOptions Options { get; set; }
        public bool IsStopRequest { get; set; }
        public bool IsStartRequest { get; set; }
    }


    public class PythonResponseMessage : IPythonMessage
    {
        [JsonIgnore]
        public List<Tensor<float>> Tensors { get; set; }
    }
}
