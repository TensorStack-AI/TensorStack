// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.TextGeneration.Common
{
    public enum EarlyStopping
    {
        /// <summary>
        /// Return all completed beams
        /// </summary>
        None = 0,

        /// <summary>
        /// Stop processing and return all completed beams when a best best has been found.
        /// </summary>
        BestBeam = 1,

        /// <summary>
        /// Stop processing and return all completed beams when {n} beams are completed.
        /// </summary>
        BeamCount = 2
    }
}
