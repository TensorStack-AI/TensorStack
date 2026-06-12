using TensorStack.Common;

namespace Amuse.App.Common
{
    public interface IDownloadModel
    {
        int Id { get; set; }
        BackendType Backend { get; set; }
        string Name { get; set; }
        ModelStatusType Status { get; set; }
        string AccessToken { get; set; }
        string Link { get; set; }

        void Initialize(Settings settings);
        void Delete(Settings settings);
        string GetDirectory(Settings settings);
    }
}
