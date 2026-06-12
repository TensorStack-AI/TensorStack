// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Amuse.App.Common;
using Amuse.App.Views;
using System.Threading.Tasks;
using TensorStack.WPF;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace Amuse.App.Dialogs
{
    /// <summary>
    /// Interaction logic for QuestionDialog.xaml
    /// </summary>
    public partial class QuestionDialog : DialogControl
    {
        private string _buttonTextSave;
        private string _buttonTextCancel;
        private string _messageTitle;
        private string _messageBody;

        public QuestionDialog()
        {
            SaveCommand = new AsyncRelayCommand(SaveAsync, CanExecuteSave);
            CancelCommand = new AsyncRelayCommand(CancelAsync, CanExecuteCancel);
            InitializeComponent();
        }

        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }

        public string ButtonTextSave
        {
            get { return _buttonTextSave; }
            set { SetProperty(ref _buttonTextSave, value); }
        }

        public string ButtonTextCancel
        {
            get { return _buttonTextCancel; }
            set { SetProperty(ref _buttonTextCancel, value); }
        }

        public string MessageTitle
        {
            get { return _messageTitle; }
            set { SetProperty(ref _messageTitle, value); }
        }

        public string MessageBody
        {
            get { return _messageBody; }
            set { SetProperty(ref _messageBody, value); }
        }


        public Task<bool> ShowDialogAsync(string title, string messageTitle, string messageBody, string buttonTextSave = "Save", string buttonTextCancel = "Cancel")
        {
            Title = title;
            MessageTitle = messageTitle;
            MessageBody = messageBody;
            ButtonTextSave = buttonTextSave;
            ButtonTextCancel = buttonTextCancel;
            return base.ShowDialogAsync();
        }
    }
}
