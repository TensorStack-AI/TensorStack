// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Windows;
using System.Windows.Controls;

namespace TensorStack.WPF.Controls
{
    public class ViewContainer : Frame
    {
        public ViewContainer()
        {
            SandboxExternalContent = false;
            NavigationUIVisibility = System.Windows.Navigation.NavigationUIVisibility.Hidden;
        }

        public static readonly DependencyProperty NavigationProperty =
             DependencyProperty.Register(nameof(Navigation), typeof(Services.NavigationService), typeof(ViewContainer),
             new PropertyMetadata<ViewContainer, Services.NavigationService>((x, o) => o.RegisterAsync(x)));

        public Services.NavigationService Navigation
        {
            get { return (Services.NavigationService)GetValue(NavigationProperty); }
            set { SetValue(NavigationProperty, value); }
        }

        public override bool ShouldSerializeContent()
        {
            return false;
        }
    }
}
