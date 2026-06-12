using System.Diagnostics;

namespace TensorStack.WPF
{
    public static class URL
    {
        public static void NavigateToUrl(string url)
        {
            System.Diagnostics.Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
        }
    }
}
