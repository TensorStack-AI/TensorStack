using System;
using System.Collections;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for NullableComboBox.xaml
    /// </summary>
    public partial class NullableComboBox : BaseControl
    {
        private NullableComboBoxItem _internalItem;

        public NullableComboBox()
        {
            InternalItems = new ObservableCollection<NullableComboBoxItem>();
            InitializeComponent();
        }

        public static readonly DependencyProperty ItemsSourceProperty = DependencyProperty.Register(nameof(ItemsSource), typeof(IEnumerable), typeof(NullableComboBox), new PropertyMetadata<NullableComboBox>((c) => c.OnItemSourceChanged()));
        public static readonly DependencyProperty SelectedItemProperty = DependencyProperty.Register(nameof(SelectedItem), typeof(object), typeof(NullableComboBox), new PropertyMetadata<NullableComboBox>((c) => c.OnSelectedItemChanged()) { BindsTwoWayByDefault = true });

        public string NullPrefix { get; set; }
        public string DisplayMemberPath { get; set; }
        public ObservableCollection<NullableComboBoxItem> InternalItems { get; }

        public IEnumerable ItemsSource
        {
            get { return (IEnumerable)GetValue(ItemsSourceProperty); }
            set { SetValue(ItemsSourceProperty, value); }
        }

        public object SelectedItem
        {
            get { return (object)GetValue(SelectedItemProperty); }
            set { SetValue(SelectedItemProperty, value); }
        }

        public NullableComboBoxItem InternalItem
        {
            get { return _internalItem; }
            set
            {
                if (SelectedItem is null && value is null)
                    return;

                if (SetProperty(ref _internalItem, value))
                {
                    SelectedItem = _internalItem?.Item;
                }
            }
        }


        private Task OnItemSourceChanged()
        {
            InternalItems.Clear();
            if (ItemsSource is null)
                return Task.CompletedTask;

            InternalItems.Add(new NullableComboBoxItem { Name = NullPrefix });
            foreach (var item in ItemsSource)
            {
                InternalItems.Add(new NullableComboBoxItem
                {
                    Name = GetDisplayValue(item, DisplayMemberPath),
                    Item = item
                });
            }
            return Task.CompletedTask;
        }


        private Task OnSelectedItemChanged()
        {
            InternalItem = InternalItems.FirstOrDefault(x => x.Item == SelectedItem)
                        ?? InternalItems.FirstOrDefault(x => x.Name == GetDisplayValue(SelectedItem, DisplayMemberPath));
            return Task.CompletedTask;
        }


        static string GetDisplayValue(object item, string displayMemberPath)
        {
            if (item is null)
                return string.Empty;

            if (item is Enum enumValue)
                return enumValue.GetDisplayName();

            if (string.IsNullOrEmpty(displayMemberPath))
                return item.ToString();

            var prop = item.GetType().GetProperty(displayMemberPath);
            return prop?.GetValue(item)?.ToString() ?? item.ToString();
        }
    }

    public class NullableComboBoxItem
    {
        public string Name { get; set; }
        public object Item { get; set; }
    }
}
