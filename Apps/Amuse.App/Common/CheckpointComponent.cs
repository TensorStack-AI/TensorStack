using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;
using TensorStack.Common;
using TensorStack.Common.Common;
using TensorStack.WPF;

namespace Amuse.App.Common
{
    public sealed class CheckpointComponent : BaseModel
    {
        private string _name;
        private CheckpointType _type;
        private string _path;
        private string _folder;
        private string[] _downloadFiles;

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool AutoMap { get; init; }

        public string Name
        {
            get { return _name; }
            set { SetProperty(ref _name, value); }
        }

        public CheckpointType Type
        {
            get { return _type; }
            set { SetProperty(ref _type, value); }
        }


        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string Path
        {
            get { return _path; }
            set { SetProperty(ref _path, value); }
        }


        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string Folder
        {
            get
            {
                if (_type == CheckpointType.LocalFile || _type == CheckpointType.LocalFolder || _type == CheckpointType.Component)
                    return null;

                return _folder;
            }
            set { SetProperty(ref _folder, value); }
        }


        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string[] DownloadFiles
        {
            get
            {
                if (_type == CheckpointType.LocalFile || _type == CheckpointType.LocalFolder || _type == CheckpointType.Component)
                    return null;

                return _downloadFiles;
            }
            set { SetProperty(ref _downloadFiles, value); }
        }


        public bool IsValid()
        {
            return IsValid(out _);
        }


        public bool IsValid(out string validation)
        {
            if (_type == CheckpointType.OnlineFolder)
            {
                if (_downloadFiles.IsNullOrEmpty())
                {
                    validation = "No download files found";
                    return false;
                }
            }
            else if (_type == CheckpointType.OnlineFile)
            {
                if (_downloadFiles.IsNullOrEmpty())
                {
                    validation = "Download file not found";
                    return false;
                }
            }
            else if (_type == CheckpointType.LocalFolder)
            {
                if (!System.IO.Directory.Exists(_path))
                {
                    validation = "Local folder not found";
                    return false;
                }
            }
            else if (_type == CheckpointType.LocalFile)
            {
                if (!System.IO.File.Exists(_path))
                {
                    validation = "Local file not found";
                    return false;
                }
            }
            else if (_type == CheckpointType.Component)
            {
                if (string.IsNullOrEmpty(_path))
                {
                    validation = "Component not selected";
                    return false;
                }
            }
            validation = null;
            return true;
        }


        public string Resolve(Settings settings, string modelDirectory)
        {
            var directory = modelDirectory ?? settings.DirectoryModel;
            if (_type == CheckpointType.OnlineFolder)
            {
                return GetSafePath(directory, _folder, _path);
            }
            if (_type == CheckpointType.OnlineFile)
            {
                var path = GetSafePath(directory, _folder, _path);
                var modelFiles = FileHelper.GetFileMapping(DownloadFiles, path);
                if (modelFiles.IsNullOrEmpty())
                    throw new System.Exception($"Checkpoint could not resolve online file '{_path}'");

                return modelFiles.Values.FirstOrDefault();
            }
            if (_type == CheckpointType.Component)
            {
                var component = settings.Components?.FirstOrDefault(x => x.Key == _path);
                if (component == null)
                    throw new System.Exception($"Checkpoint could not resolve component '{_path}'");

                return component.Checkpoint.Resolve(settings, null);
            }
            return _path;
        }


        public bool IsInstalled(string modelDirectory, IReadOnlyCollection<ComponentModel> components = default)
        {
            if (_type == CheckpointType.OnlineFolder)
            {
                var path = GetSafePath(modelDirectory, _folder, _path);
                var modelFiles = AutoMap ? FileHelper.GetUrlFileMapping(DownloadFiles, path) : FileHelper.GetFileMapping(DownloadFiles, path);
                if (modelFiles.Values.All(System.IO.File.Exists))
                {
                    return true;
                }
            }
            if (_type == CheckpointType.OnlineFile)
            {
                var path = GetSafePath(modelDirectory, _folder, _path);
                var modelFiles = FileHelper.GetFileMapping(DownloadFiles, path);
                if (modelFiles.Values.All(System.IO.File.Exists))
                {
                    return true;
                }
            }
            else if (_type == CheckpointType.LocalFolder)
            {
                return System.IO.Directory.Exists(_path);
            }
            else if (_type == CheckpointType.LocalFile)
            {
                return System.IO.File.Exists(_path);
            }
            else if (_type == CheckpointType.Component)
            {
                var component = components?.FirstOrDefault(x => x.Key == _path);
                if (component == null)
                    return false;

                return component.Status == ModelStatusType.Installed;
            }
            return false;
        }


        public CheckpointComponent DeepClone()
        {
            return new CheckpointComponent
            {
                Name = Name,
                Type = Type,
                Folder = Folder,
                Path = Path,
                AutoMap = AutoMap,
                DownloadFiles = DownloadFiles?.ToArray(),
            };
        }


        public static string GetSafePath(string directory, string folder, string path)
        {
            if (!string.IsNullOrEmpty(folder))
            {
                return string.IsNullOrWhiteSpace(path)
                     ? System.IO.Path.Combine(directory, folder)
                     : System.IO.Path.Combine(directory, folder, path);
            }

            return string.IsNullOrWhiteSpace(path)
                  ? directory
                  : System.IO.Path.Combine(directory, path);
        }
    }
}
