// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.WPF.Controls;
using System.Threading.Tasks;

namespace TensorStack.WPF.Dialogs
{
    /// <summary>
    /// Interaction logic for MessageDialog.xaml
    /// </summary>
    public partial class MessageDialog : DialogControl
    {
        private string _message;
        private MessageDialogType _dialogType;
        private MessageBoxIconType _messageBoxIcon;
        private MessageBoxStyleType _messageBoxStyle;
        private bool _isDontAskEnabled;
        private bool _isDontAskSelected;

        public MessageDialog() 
        {
            OkCommand = new AsyncRelayCommand(Ok);
            NoCommand = new AsyncRelayCommand(No);
            YesCommand = new AsyncRelayCommand(Yes);
            InitializeComponent();
        }

        public AsyncRelayCommand OkCommand { get; }
        public AsyncRelayCommand NoCommand { get; }
        public AsyncRelayCommand YesCommand { get; }

        public string Message
        {
            get { return _message; }
            set { _message = value; NotifyPropertyChanged(); }
        }

        public MessageDialogType DialogType
        {
            get { return _dialogType; }
            set { _dialogType = value; NotifyPropertyChanged(); }
        }


        public MessageBoxIconType MessageBoxIcon
        {
            get { return _messageBoxIcon; }
            set { _messageBoxIcon = value; NotifyPropertyChanged(); }
        }


        public MessageBoxStyleType MessageBoxStyle
        {
            get { return _messageBoxStyle; }
            set { _messageBoxStyle = value; NotifyPropertyChanged(); }
        }

        public bool IsDontAskEnabled
        {
            get { return _isDontAskEnabled; }
            set { _isDontAskEnabled = value; NotifyPropertyChanged(); }
        }

        public bool IsDontAskSelected
        {
            get { return _isDontAskSelected; }
            set { _isDontAskSelected = value; NotifyPropertyChanged(); }
        }



        public Task<bool> ShowDialogAsync(string title, string message, MessageDialogType dialogType = MessageDialogType.Ok, MessageBoxIconType messageBoxIcon = MessageBoxIconType.None, MessageBoxStyleType messageBoxStyle = MessageBoxStyleType.None, bool isDontAskEnabled = false)
        {
            Title = title;
            Message = message;
            DialogType = dialogType;
            MessageBoxIcon = messageBoxIcon;
            MessageBoxStyle = messageBoxStyle;
            IsDontAskEnabled = isDontAskEnabled;
            return base.ShowDialogAsync();
        }


        private Task Ok()
        {
            return base.SaveAsync();
        }


        private Task No()
        {
            return base.CancelAsync();
        }


        private Task Yes()
        {
            return base.SaveAsync();
        }



    }

    public enum MessageDialogType
    {
        Ok,
        YesNo
    }

    public enum MessageBoxIconType
    {
        None = 0,
        Question = 1,
        Info = 2,
        Warning = 3,
        Error = 4,
    }

    public enum MessageBoxStyleType
    {
        None = 0,
        Info = 1,
        Warning = 2,
        Success = 3,
        Danger = 4,
    }
}
