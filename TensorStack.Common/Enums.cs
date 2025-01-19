// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.

namespace TensorStack.Common
{
    /// <summary>
    /// Enum Optimization
    /// </summary>
    public enum Optimization
    {
        /// <summary>
        /// No Optimizations
        /// </summary>
        None = 0,

        /// <summary>
        /// Basic Optimizations
        /// </summary>
        Basic = 1,

        /// <summary>
        /// Extended Optimizations
        /// </summary>
        Extended = 2,

        /// <summary>
        /// All Optimizations
        /// </summary>
        All = 99
    }


    /// <summary>
    /// Enum Provider
    /// </summary>
    public enum Provider
    {
        /// <summary>
        /// CPU provider (does not support Float16 or BFloat16)
        /// </summary>
        CPU = 0,

        /// <summary>
        /// The Microsoft DirectML Provider
        /// </summary>
        DirectML = 1,

        /// <summary>
        /// The Nividia CUDA Provider (Requires CUDA SDK/Toolkit)
        /// </summary> 
        CUDA = 2,

        /// <summary>
        /// The Apple CoreML Provider
        /// </summary>
        CoreML = 3,

        /// <summary>
        /// Custom Provider
        /// </summary>
        Custom = 100
    }


    /// <summary>
    /// Normalization
    /// </summary>
    public enum Normalization
    {
        None = 0,
        ZeroToOne = 1,
        OneToOne = 2,
        MinMax = 3
    }


    /// <summary>
    /// Enum ResizeMode
    /// </summary>
    public enum ResizeMode
    {
        /// <summary>
        /// Center Crop Image
        /// </summary>
        Crop = 0,

        /// <summary>
        /// Strech Image
        /// </summary>
        Stretch = 1
    }
}
