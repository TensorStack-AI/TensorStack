// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
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
        public Task LoadAsync();

        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        public Task UnloadAsync();
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
           where O : RunOptions
           where P : RunProgress
    {
        Task<T> RunAsync(O options, IProgress<P> progressCallback = default, CancellationToken cancellationToken = default);
    }


    /// <summary>
    /// Interface IPipeline
    /// Extends the <see cref="TensorStack.Common.Pipeline.IPipeline" />
    /// </summary>
    /// <typeparam name="T">The return type</typeparam>
    /// <typeparam name="I">The input parameter type</typeparam>
    /// <typeparam name="O">The RunOptions type</typeparam>
    /// <typeparam name="P">The RunProgress type</typeparam>
    /// <seealso cref="TensorStack.Common.Pipeline.IPipeline" />
    public interface IPipeline<T, I, O, P> : IPipeline
       where T : class
       where I : class
       where O : RunOptions
       where P : RunProgress
    {
        Task<T> RunAsync(O options, I input, IProgress<P> progressCallback = default, CancellationToken cancellationToken = default);
    }
}
