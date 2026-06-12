// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Amuse.App.Common;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.WPF;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace Amuse.App.Dialogs
{
    /// <summary>
    /// Interaction logic for LoraModelDialog.xaml
    /// </summary>
    public partial class LoraModelDialog : DialogControl
    {
        private LoraAdapterModel _loraModel;
        private LoraAdapterModel _originalLoraModel;
        private string _selectedTrigger;

        public LoraModelDialog(Settings settings)
        {
            Settings = settings;
            Trigger = new ObservableCollection<string>();
            SaveCommand = new AsyncRelayCommand(SaveAsync, CanExecuteSave);
            CancelCommand = new AsyncRelayCommand(CancelAsync);
            AddTriggerCommand = new AsyncRelayCommand(AddTriggerAsync, CanAddTrigger);
            RemoveTriggerCommand = new AsyncRelayCommand<string>(RemoveTriggerAsync);
            Errors = new ObservableCollection<string>();
            InitializeComponent();
        }

        public Settings Settings { get; }
        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }
        public ObservableCollection<string> Errors { get; }
        public AsyncRelayCommand AddTriggerCommand { get; }
        public AsyncRelayCommand<string> RemoveTriggerCommand { get; }
        public ObservableCollection<string> Trigger { get; }
        public CheckpointType[] CheckpointTypes { get; } = [CheckpointType.OnlineFile, CheckpointType.LocalFile];
        public bool IsUpdateMode => _originalLoraModel is not null;

        public LoraAdapterModel LoraModel
        {
            get { return _loraModel; }
            set { SetProperty(ref _loraModel, value); }
        }

        public string SelectedTrigger
        {
            get { return _selectedTrigger; }
            set { SetProperty(ref _selectedTrigger, value); }
        }

        public Task<bool> AddAsync()
        {
            var modelId = GetNextModelId();
            LoraModel = new LoraAdapterModel
            {
                Id = modelId,
                Backend = BackendType.PyTorch,
                Pipeline = Settings.DiffusionPipelines.First(),
                Name = "New Lora",
                Checkpoint = new CheckpointComponent
                {
                    Name = "LoraAdapter",
                    Type = CheckpointType.LocalFile
                }
            };
            return base.ShowDialogAsync();
        }


        public Task<bool> UpdateAsync(LoraAdapterModel loraModel)
        {
            var modelId = loraModel.Id;
            _originalLoraModel = loraModel;
            LoraModel = loraModel.DeepClone(modelId);
            Populate();
            return base.ShowDialogAsync();
        }


        public Task<bool> CopyAsync(LoraAdapterModel loraModel)
        {
            var modelId = GetNextModelId();
            LoraModel = loraModel.DeepClone(modelId);
            LoraModel.Name += " copy";
            Populate();
            return base.ShowDialogAsync();
        }


        public async Task<bool> ImportAsync(LoraAdapterModel[] modelImports)
        {
            var modelId = GetNextModelId();
            if (modelImports.Length == 1)
            {
                var modelImport = modelImports[0];
                modelImport.Id = modelId;
                LoraModel = modelImport;
                Populate();
                return await base.ShowDialogAsync();
            }
            else
            {
                var imported = 0;
                foreach (var modelImport in modelImports)
                {
                    if (Settings.LoraAdapterModels.Any(x => x.Backend == modelImport.Backend && x.Name == modelImport.Name && x.Pipeline == modelImport.Pipeline))
                        continue;

                    imported++;
                    modelImport.Id = modelId++;
                    Settings.LoraAdapterModels.Add(modelImport);
                }

                await DialogService.ShowMessageAsync("Import Complete", $"{imported}/{modelImports.Length} Lora Adapters Imported.");
                return true;
            }
        }


        protected override Task SaveAsync()
        {
            var index = Settings.LoraAdapterModels.Count;
            if (IsUpdateMode)
            {
                index = Settings.LoraAdapterModels.IndexOf(_originalLoraModel);
                Settings.LoraAdapterModels.Remove(_originalLoraModel);
            }

            LoraModel.Key = CreateKey();
            LoraModel.Triggers = Trigger.Count == 0 ? default : Trigger.ToArray();
            Settings.LoraAdapterModels.Insert(index, LoraModel);
            return base.SaveAsync();
        }


        protected override bool CanExecuteSave()
        {
            if (LoraModel == null)
                return false;

            Errors.Clear();
            foreach (var inputError in GetValidationErrors())
                Errors.Add(inputError);

            return Errors.Count == 0 && base.CanExecuteSave();
        }


        protected override Task CancelAsync()
        {
            LoraModel = default;
            _originalLoraModel = null;
            return base.CancelAsync();
        }


        protected override async Task CloseAsync()
        {
            await base.CloseAsync();
        }


        private Task AddTriggerAsync()
        {
            Trigger.Add(_selectedTrigger);
            SelectedTrigger = null;
            return Task.CompletedTask;
        }


        private bool CanAddTrigger()
        {
            return !Trigger.Contains(_selectedTrigger);
        }


        private Task RemoveTriggerAsync(string trigger)
        {
            Trigger.Remove(trigger);
            SelectedTrigger = null;
            return Task.CompletedTask;
        }


        private int GetNextModelId()
        {
            return Math.Max(Utils.FixedIdRange, Settings.LoraAdapterModels.Max(x => x.Id)) + 1;
        }


        private void Populate()
        {
            if (!LoraModel.Triggers.IsNullOrEmpty())
            {
                foreach (var trigger in LoraModel.Triggers)
                {
                    Trigger.Add(trigger);
                }
            }
        }


        private IEnumerable<string> GetValidationErrors()
        {
            // Name
            if (string.IsNullOrWhiteSpace(LoraModel.Name))
                yield return "Name cannot be empty";
            if (!IsUpdateMode)
            {
                if (Settings.LoraAdapterModels.Any(x => x.Name.Equals(LoraModel.Name, StringComparison.OrdinalIgnoreCase)))
                    yield return $"Model with Name '{LoraModel.Name}' already exists";
            }

            // Checkpoint
            if (!LoraModel.Checkpoint.IsValid(out var checkpointValidation))
                yield return checkpointValidation;
        }


        private string CreateKey()
        {
            return $"{new string([.. LoraModel.Name.Where(char.IsLetterOrDigit)])}{LoraModel.Id}".ToLower();
        }

    }
}
