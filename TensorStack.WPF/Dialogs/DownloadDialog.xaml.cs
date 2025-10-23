// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.ComponentModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.WPF.Dialogs
{
    /// <summary>
    /// Interaction logic for DownloadDialog.xaml
    /// </summary>
    public partial class DownloadDialog : DialogControl
    {
        private readonly DownloadService _downloadService;
        private readonly Progress<DownloadProgress> _downloadCallback;
        private string _message;
        private double _progress;
        private string _downloadSource;
        private string[] _downloadSources;
        private string _downloadDestination;
        private double _speed;
        private double _totalSize;
        private double _totalDownloaded;
        private string _cancelText = "No";
        private CancellationTokenSource _cancellationTokenSource;
        private DateTime _lastSpeedUpdate;
        private DateTime _lastProgessUpdate;

        public DownloadDialog(DownloadService downloadService)
        {
            _downloadService = downloadService;
            _downloadCallback = new Progress<DownloadProgress>(OnDownloadProgress);
            NoCommand = new AsyncRelayCommand(CloseAsync);
            YesCommand = new AsyncRelayCommand(Yes);
            InitializeComponent();
        }

        public AsyncRelayCommand NoCommand { get; }
        public AsyncRelayCommand YesCommand { get; }

        public string Message
        {
            get { return _message; }
            set { SetProperty(ref _message, value); }
        }

        public double Progress
        {
            get { return _progress; }
            set { SetProperty(ref _progress, value); }
        }

        public double Speed
        {
            get { return _speed; }
            set { SetProperty(ref _speed, value); }
        }

        public double TotalSize
        {
            get { return _totalSize; }
            set { SetProperty(ref _totalSize, value); }
        }

        public double TotalDownloaded
        {
            get { return _totalDownloaded; }
            set { SetProperty(ref _totalDownloaded, value); }
        }

        public string CancelText
        {
            get { return _cancelText; }
            set { SetProperty(ref _cancelText, value); }
        }


        public Task<bool> ShowDialogAsync(string message, string downloadSource, string downloadDestination)
        {
            Progress = 0;
            Title = "Download";
            Message = message;
            _downloadSource = downloadSource;
            _downloadDestination = downloadDestination;
            return base.ShowDialogAsync();
        }


        public Task<bool> ShowDialogAsync(string message, string[] downloadSources, string downloadDestination)
        {
            Progress = 0;
            Title = "Download";
            Message = message;
            _downloadSources = downloadSources;
            _downloadDestination = downloadDestination;
            return base.ShowDialogAsync();
        }


        private async Task Yes()
        {
            try
            {
                CancelText = "Cancel";
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    if (!_downloadSources.IsNullOrEmpty())
                    {
                        await _downloadService.DownloadAsync([.. _downloadSources], _downloadDestination, _downloadCallback, _cancellationTokenSource.Token);
                    }
                    else
                    {
                        await _downloadService.DownloadAsync(_downloadSource, _downloadDestination, _downloadCallback, _cancellationTokenSource.Token);
                    }

                    await base.SaveAsync();
                }
            }
            catch (OperationCanceledException)
            {
            }
        }


        protected override async Task CloseAsync()
        {
            await _cancellationTokenSource.SafeCancelAsync();
            await base.CloseAsync();
        }


        private void OnDownloadProgress(DownloadProgress progress)
        {
            Progress = progress.TotalProgress;
            TotalSize = progress.TotalSize / 1024.0 / 1024.0;

            if (DateTime.UtcNow > _lastSpeedUpdate)
            {
                _lastSpeedUpdate = DateTime.UtcNow.AddMilliseconds(1000);
                Speed = progress.BytesSec / 1024.0 / 1024.0;
            }

            if (DateTime.UtcNow > _lastProgessUpdate)
            {
                _lastProgessUpdate = DateTime.UtcNow.AddMilliseconds(250);
                TotalDownloaded = progress.TotalBytes / 1024.0 / 1024.0;
            }
        }

    }
}
