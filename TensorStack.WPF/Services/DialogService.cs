// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.WPF.Controls;
using TensorStack.WPF.Dialogs;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Win32;
using System;
using System.IO;
using System.Threading.Tasks;

namespace TensorStack.WPF.Services
{
    public sealed class DialogService
    {
        private static DialogFactory _dialogFactory;

        public DialogService(IServiceProvider serviceProvider)
        {
            if (_dialogFactory != null)
                throw new Exception("DialogService already initialized");

            _dialogFactory = new DialogFactory(serviceProvider);
        }

        public static T GetDialog<T>() where T : DialogControl
        {
            return GetDialogInstance<T>();
        }


        public static async Task<bool> ShowErrorAsync(string title, string message)
        {
            var dialog = GetDialogInstance<MessageDialog>();
            var result = await dialog.ShowDialogAsync(title, message, MessageDialogType.Ok, MessageBoxIconType.Error, MessageBoxStyleType.Danger);
            return new DialogResult(result, false);
        }

        public static async Task<DialogResult> ShowMessageAsync(string title, string message, MessageDialogType dialogType = MessageDialogType.Ok, MessageBoxIconType messageBoxIcon = MessageBoxIconType.None, MessageBoxStyleType messageBoxStyle = MessageBoxStyleType.None, bool isDontAskEnabled = false)
        {
            var dialog = GetDialogInstance<MessageDialog>();
            var result = await dialog.ShowDialogAsync(title, message, dialogType, messageBoxIcon, messageBoxStyle, isDontAskEnabled);
            return new DialogResult(result, dialog.IsDontAskSelected);
        }

        public static Task<string> OpenFolderAsync(string title, string initialDirectory = default)
        {
            var folderBrowserDialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = title,
                InitialDirectory = initialDirectory,
                UseDescriptionForTitle = true,
                AutoUpgradeEnabled = true
            };

            var ownerWindow = _dialogFactory.GetOwner<WindowMainBase>();
            var dialogResult = folderBrowserDialog.ShowDialog(ownerWindow);
            if (dialogResult == System.Windows.Forms.DialogResult.OK)
                return Task.FromResult(folderBrowserDialog.SelectedPath);

            return Task.FromResult<string>(default);
        }


        public static Task<string> SaveFileAsync(string title, string initialFilename, string initialDirectory = default, string filter = default, string defualtExt = default)
        {
            var saveFileDialog = new SaveFileDialog
            {
                Title = title,
                Filter = filter,
                DefaultExt = defualtExt,
                AddExtension = true,
                RestoreDirectory = string.IsNullOrEmpty(initialDirectory),
                InitialDirectory = initialDirectory,
                FileName = initialFilename
            };

            var ownerWindow = _dialogFactory.GetOwner<WindowMainBase>();
            var dialogResult = saveFileDialog.ShowDialog(ownerWindow);
            if (dialogResult == true)
                return Task.FromResult(saveFileDialog.FileName);

            return Task.FromResult<string>(default);
        }


        public static Task<string> OpenFileAsync(string title, string initialDirectory = default, string filter = default, string defualtExt = default)
        {
            if (Path.HasExtension(initialDirectory))
                initialDirectory = Path.GetDirectoryName(initialDirectory);

            var openFileDialog = new OpenFileDialog
            {
                Title = title,
                Filter = filter,
                CheckFileExists = true,
                InitialDirectory = initialDirectory,
                RestoreDirectory = string.IsNullOrEmpty(initialDirectory),
                DefaultExt = defualtExt,
                AddExtension = true,
                Multiselect = false
            };

            var ownerWindow = _dialogFactory.GetOwner<WindowMainBase>();
            var dialogResult = openFileDialog.ShowDialog(ownerWindow);
            if (dialogResult == true)
                return Task.FromResult(openFileDialog.FileName);

            return Task.FromResult<string>(default);
        }


        public static async Task<DialogResult> DownloadAsync(string message,  string downloadSource, string downloadDestination)
        {
            var dialog = GetDialogInstance<DownloadDialog>();
            var result = await dialog.ShowDialogAsync(message, downloadSource, downloadDestination);
            return new DialogResult(result, false);
        }


        public static async Task<DialogResult> DownloadAsync(string message, string[] downloadSource, string downloadDestination)
        {
            var dialog = GetDialogInstance<DownloadDialog>();
            var result = await dialog.ShowDialogAsync(message, downloadSource, downloadDestination);
            return new DialogResult(result, false);
        }


        private static T GetDialogInstance<T>() where T : DialogControl
        {
            return _dialogFactory.CreateDialog<T, WindowMainBase>();
        }
    }


    public sealed class DialogFactory
    {
        private readonly IServiceProvider _serviceProvider;

        public DialogFactory(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }

        public T GetOwner<T>() where T : WindowMainBase
        {
            return _serviceProvider.GetRequiredService<T>();
        }

        public T CreateDialog<T>() where T : DialogControl
        {
            return _serviceProvider.GetRequiredService<T>();
        }

        public T CreateDialog<T, U>() where T : DialogControl where U : WindowMainBase
        {
            var dialog = CreateDialog<T>();
            dialog.Owner = GetOwner<U>();
            return dialog;
        }
    }
}
