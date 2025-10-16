// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.WPF.Controls;
using Microsoft.Extensions.DependencyInjection;
using System;

namespace TensorStack.WPF.Services
{
    public class ComponentService
    {
        private IServiceProvider _serviceProvider;

        public ComponentService(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }

        public T GetComponent<T>() where T : Component
        {
            return _serviceProvider.GetService<T>();
        }
    }
}
