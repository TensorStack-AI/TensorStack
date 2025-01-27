// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using TensorStack.Common.Tensor;

namespace TensorStack.Common.Image
{
    public abstract class ImageInput<T> : ImageTensor where T : class
    {
        public ImageInput(Tensor<float> tensor) 
            : base(tensor) { }

        public ImageInput(ReadOnlySpan<int> dimensions)
            : base(dimensions) { }

        public abstract T Image { get; }
        public abstract void Resize(int width, int height, ResizeMode resizeMode = ResizeMode.Stretch);
        public abstract void Save(string filename);
    }
}
