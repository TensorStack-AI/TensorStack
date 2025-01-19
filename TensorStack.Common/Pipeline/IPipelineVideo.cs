// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Common.Pipeline
{
    /// <summary>
    /// Video Pipeline Interface
    /// Extends the <see cref="TensorStack.Common.Pipeline.IPipeline" />
    /// </summary>
    /// <typeparam name="O">The RunOptions type</typeparam>
    /// <typeparam name="P">The RunProgress type</typeparam>
    /// <seealso cref="TensorStack.Common.Pipeline.IPipeline" />
    public interface IPipelineVideo<O, P> : IPipeline
        where O : RunOptions
        where P : RunProgress
    {

        /// <summary>
        /// Runs pipeline with the specified options
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputVideo">The input video.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>Task&lt;VideoTensor&gt;.</returns>
        Task<VideoTensor> RunVideoAsync(O options, VideoTensor inputVideo, IProgress<P> progressCallback = default, CancellationToken cancellationToken = default);

        /// <summary>
        /// Gets the video pipeline stream
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputVideoStream">The input video stream.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>IAsyncEnumerable&lt;VideoFrame&gt;.</returns>
        IAsyncEnumerable<VideoFrame> GetStreamAsync(O options, IAsyncEnumerable<VideoFrame> videoStream, IProgress<P> progressCallback = default, CancellationToken cancellationToken = default);
    }


    /// <summary>
    /// Video Pipeline Interface with default RunProgress
    /// Extends the <see cref="TensorStack.Common.Pipeline.IPipelineVideo{O, TensorStack.Common.Pipeline.RunProgress}" />
    /// </summary>
    /// <typeparam name="O">The RunOptions type</typeparam>
    /// <seealso cref="TensorStack.Common.Pipeline.IPipelineVideo{O, TensorStack.Common.Pipeline.RunProgress}" />
    public interface IPipelineVideo<O> : IPipelineVideo<O, RunProgress> 
        where O : RunOptions { }
}
