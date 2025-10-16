// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.WPF.Controls;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace TensorStack.WPF.Services
{
    public class NavigationService
    {
        private ViewContainer _host;
        private List<IViewControl> _views;
        private IServiceProvider _serviceProvider;

        public NavigationService(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }

        public IReadOnlyList<IViewControl> Views => _views;
        public IViewControl CurrentView => _host?.Content as IViewControl;

        public Task RegisterAsync(ViewContainer host)
        {
            _host = host;
            _host.JournalOwnership = System.Windows.Navigation.JournalOwnership.OwnsJournal;
            _views = new List<IViewControl>(_serviceProvider.GetServices<IViewControl>());
            return Task.CompletedTask;
        }

        public Task NavigateAsync(int view)
        {
            return NavigateAsync<OpenViewArgs>(view);
        }


        public Task NavigateAsync(Type type)
        {
            return NavigateAsync<OpenViewArgs>(type);
        }


        public async Task NavigateAsync<U>(int view, U openViewArgs = default) where U : OpenViewArgs
        {
            var viewControl = _views.First(x => x.Id == view);
            await HostNavigate(viewControl, openViewArgs);
        }


        public async Task NavigateAsync<U>(Type type, U openViewArgs = default) where U : OpenViewArgs
        {
            var viewControl = _views.First(x => x.GetType() == type);
            await HostNavigate(viewControl, openViewArgs);
        }


        private async Task HostNavigate<U>(IViewControl viewControl, U openViewArgs) where U : OpenViewArgs
        {
            if (CurrentView != null)
                await CurrentView.CloseAsync();

            var clonedOpenViewArgs = CloneOpenViewArgs(openViewArgs);

            _host.Navigate(viewControl);
            _host.RemoveBackEntry();
            await viewControl.OpenAsync(clonedOpenViewArgs);
        }


        private U CloneOpenViewArgs<U>(U data) where U : OpenViewArgs
        {
            if (data == null)
                return default;
            return Serializer.DeepClone(data);
        }
    }
}
