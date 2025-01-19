// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.Common
{
    public record ModelConfig(string Path, Provider Provider, int DeviceId, bool IsOptimizationSupported);

}
