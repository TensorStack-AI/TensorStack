// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using TensorStack.StableDiffusion.Common;

namespace TensorStack.StableDiffusion
{
    public static class Extensions
    {
        /// <summary>
        /// Notifies the specified message.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="message">The message.</param>
        public static void Notify(this IProgress<GenerateProgress> progressCallback, string message)
        {
            progressCallback?.Report(new GenerateProgress(message));
        }
    }
}
