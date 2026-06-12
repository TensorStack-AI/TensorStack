using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Controls;

namespace TensorStack.WPF.Controls
{
    public class Component : UserControl, INotifyPropertyChanged
    {
        public Component()
        {

        }

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }
}
