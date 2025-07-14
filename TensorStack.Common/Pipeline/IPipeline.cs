// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TensorStack.Common.Pipeline
{
    /// <summary>
    /// Basic IPipeline Interface
    /// Extends the <see cref="IDisposable" />
    /// </summary>
    /// <seealso cref="IDisposable" />
    public interface IPipeline : IDisposable
    {
        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        public Task LoadAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        public Task UnloadAsync(CancellationToken cancellationToken = default);
    }


    /// <summary>
    /// Interface IPipeline
    /// Extends the <see cref="TensorStack.Common.Pipeline.IPipeline" />
    /// </summary>
    /// <typeparam name="T">The return type</typeparam>
    /// <typeparam name="O">The RunOptions type</typeparam>
    /// <typeparam name="P">The RunProgress type</typeparam>
    /// <seealso cref="TensorStack.Common.Pipeline.IPipeline" />
    public interface IPipeline<T, O, P> : IPipeline
        where T : class
        where O : IRunOptions
        where P : IRunProgress
    {
        Task<T> RunAsync(O options, IProgress<P> progressCallback = default, CancellationToken cancellationToken = default);
    }


    /// <summary>
    /// Interface IPipeline
    /// Extends the <see cref="TensorStack.Common.Pipeline.IPipeline" />
    /// </summary>
    /// <typeparam name="T">The return type</typeparam>
    /// <typeparam name="O">The RunOptions type</typeparam>
    /// <seealso cref="TensorStack.Common.Pipeline.IPipeline" />
    public interface IPipeline<T, O> : IPipeline<T, O, RunProgress>
        where T : class
        where O : IRunOptions
    {
    }


    /// <summary>Interface IPipelineStream
    /// Extends the <see cref="T:TensorStack.Common.Pipeline.IPipeline" /></summary>
    /// <typeparam name="T">Result type</typeparam>
    /// <typeparam name="O">The RunOptions type</typeparam>
    /// <typeparam name="P">The RunProgress type</typeparam>
    public interface IPipelineStream<T, O, P> : IPipeline
        where T : class
        where O : IRunOptions
        where P : IRunProgress
    {
        IAsyncEnumerable<T> RunAsync(O options, IProgress<P> progressCallback = default, CancellationToken cancellationToken = default);
    }


    /// <summary>
    /// Interface IPipelineStream
    /// Extends the <see cref="TensorStack.Common.Pipeline.IPipelineStream{T, O, TensorStack.Common.Pipeline.RunProgress}" />
    /// </summary>
    /// <typeparam name="T">Result type</typeparam>
    /// <typeparam name="O">The RunOptions type</typeparam>
    /// <seealso cref="TensorStack.Common.Pipeline.IPipelineStream{T, O, TensorStack.Common.Pipeline.RunProgress}" />
    public interface IPipelineStream<T, O> : IPipelineStream<T, O, RunProgress>
        where T : class
        where O : IRunOptions
    { 
    }
}
