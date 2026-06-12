// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;

namespace Amuse.App.Common
{
    public sealed class WizardItemModel
    {
        public string Name { get; set; }
        public BackendType Backend { get; set; }
        public WizardOptionModel[] Options { get; set; }
    }
}
