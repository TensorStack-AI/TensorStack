using System.Windows;
using TensorStack.WPF.Controls;

namespace TensorStack.WPF.Utils
{
    public static class DragDropHelper
    {

        public static DragDropEffects DoDragDropFile(DependencyObject dragSource, string filepath, DragDropType dropType, UIElement visual = null, double visualScale = 1)
        {
            if (Application.Current.MainWindow is WindowMainBase mainWindow)
            {
                return mainWindow.DoDragDropFile(dragSource, filepath, dropType, visual, visualScale);
            }

            throw new System.Exception("WindowMainBase not found");
        }

        public static DragDropEffects DoDragDropObject<T>(DependencyObject dragSource, T dropData, DragDropType dropType, UIElement visual = null, double visualScale = 1)
        {
            if (Application.Current.MainWindow is WindowMainBase mainWindow)
            {
                return mainWindow.DoDragDropObject<T>(dragSource, dropData, dropType, visual, visualScale);
            }

            throw new System.Exception("WindowMainBase not found");
        }
    }
}
