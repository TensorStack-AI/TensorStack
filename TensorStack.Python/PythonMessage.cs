using System.Collections.Generic;
using System.Text.Json.Serialization;
using TensorStack.Common.Tensor;
using TensorStack.Python.Common;

namespace TensorStack.Python
{
    public interface IPythonMessage
    {
        List<Tensor<float>> Tensors { get; set; }
    }

    public enum PythonMessageType
    {
        Start = 0,
        Stop = 1,
        Data = 2
    }

    public class PythonRequestMessage : IPythonMessage
    {
        public PythonRequestMessage() { }
        public PythonRequestMessage(PythonMessageType type)
        {
            Type = type;
        }

        public PythonMessageType Type { get; init; }
        public PythonOptions Options { get; set; }

        [JsonIgnore]
        public List<Tensor<float>> Tensors { get; set; }
    }


    public class PythonResponseMessage : IPythonMessage
    {
        [JsonIgnore]
        public List<Tensor<float>> Tensors { get; set; }
    }
}
