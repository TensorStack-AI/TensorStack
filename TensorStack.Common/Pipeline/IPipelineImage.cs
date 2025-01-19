// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;

namespace TensorStack.Common.Pipeline
{
    /// <summary>
    /// Image Pipeline Interface
    /// Extends the <see cref="TensorStack.Common.Pipeline.IPipeline" />
    /// </summary>
    /// <typeparam name="O">The RunOptions type</typeparam>
    /// <typeparam name="P">The RunProgress type</typeparam>
    /// <seealso cref="TensorStack.Common.Pipeline.IPipeline" />
    public interface IPipelineImage<O, P> : IPipeline
        where O : RunOptions
        where P : RunProgress
    {
        Task<ImageTensor> RunImageAsync(O options, ImageTensor inputImage, IProgress<P> progressCallback = default, CancellationToken cancellationToken = default);
    }


    /// <summary>
    /// Image Pipeline Interface with default RunProgress
    /// Extends the <see cref="TensorStack.Common.Pipeline.IPipelineImage{O, TensorStack.Common.Pipeline.RunProgress}" />
    /// </summary>
    /// <typeparam name="O">The RunOptions type</typeparam>
    /// <seealso cref="TensorStack.Common.Pipeline.IPipelineImage{O, TensorStack.Common.Pipeline.RunProgress}" />
    public interface IPipelineImage<O> : IPipelineImage<O, RunProgress>
        where O : RunOptions { }
}
