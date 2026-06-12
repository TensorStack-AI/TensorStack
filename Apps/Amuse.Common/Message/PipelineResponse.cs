using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using TensorStack.Common.Tensor;

namespace Amuse.Common.Message
{
    public sealed class PipelineResponse : IPipelineMessage
    {
        public PipelineResponse() { }
        public PipelineResponse(Exception ex) : this(ex.Message)
        {
            IsCanceled = ex is OperationCanceledException;
        }
        public PipelineResponse(string errorMessage)
        {
            Error = errorMessage;
            Type = ResponseType.Error;
        }

        public PipelineResponse(params IReadOnlyList<Tensor<float>> tensors)
        {
            Tensors = tensors;
            Type = ResponseType.Tensor;
        }


        public PipelineResponse(params PipelineTextResult[] textResults)
        {
            Type = ResponseType.Object;
            TextResponse = new PipelineTextResponse(textResults);
        }

        public ResponseType Type { get; init; }
        public PipelineTextResponse TextResponse { get; set; }

        public string Error { get; init; }
        public bool IsCanceled { get; init; }

        [JsonIgnore]
        public IReadOnlyList<Tensor<float>> Tensors { get; set; }


        [JsonIgnore]
        public bool IsError => !string.IsNullOrEmpty(Error);
    }


    public sealed record PipelineTextResponse
    {
        public PipelineTextResponse() { }
        public PipelineTextResponse(PipelineTextResult[] results)
        {
            Results = results;
        }
        public PipelineTextResult[] Results { get; set; }
    }

    public sealed record PipelineTextResult
    {
        public int Beam { get; set; }
        public float Score { get; set; }
        public float PenaltyScore { get; set; }
        public string Text { get; set; }
    }
}
