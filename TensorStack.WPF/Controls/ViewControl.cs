// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Controls;
using TensorStack.WPF.Services;


namespace TensorStack.WPF.Controls
{
    public abstract class ViewControl : UserControl, IViewControl, INotifyPropertyChanged
    {
        private readonly NavigationService _navigationService;
        private bool _isDragDrop;
        private DragDropType _dragDropType;

        public ViewControl(NavigationService navigationService)
        {
            _navigationService = navigationService;
        }

        public virtual int Id { get; }
        public NavigationService NavigationService => _navigationService;

        public bool IsDragDrop
        {
            get { return _isDragDrop; }
            set { SetProperty(ref _isDragDrop, value); }
        }

        public DragDropType DragDropType
        {
            get { return _dragDropType; }
            set { SetProperty(ref _dragDropType, value); }
        }

        public virtual Task OpenAsync(OpenViewArgs args = default)
        {
            return Task.CompletedTask;
        }


        public virtual Task CloseAsync()
        {
            return Task.CompletedTask;
        }


        public virtual Task<bool> ShutdownAsync(bool force = false)
        {
            return Task.FromResult(false);
        }



        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        protected void NotifyPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetProperty<T>(ref T field, T newValue, [CallerMemberName] string propertyName = null)
        {
            if (!Equals(field, newValue))
            {
                field = newValue;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
                return true;
            }

            return false;
        }
        #endregion
    }

    public interface IViewControl
    {
        int Id { get; }
        Task OpenAsync(OpenViewArgs args = default);
        Task CloseAsync();
        Task<bool> ShutdownAsync(bool force = false);
        NavigationService NavigationService { get; }
        bool IsDragDrop { get; set; }
        DragDropType DragDropType { get; set; }
    }

    public record OpenViewArgs;
}
