using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;
using TensorStack.Example.Services;
using TensorStack.WPF;

namespace TensorStack.Example
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private readonly IHost _appHost;

        public App()
        {
            RegisterExceptionHandlers();

            var builder = Host.CreateApplicationBuilder();

            var configuration = Json.Load<Settings>("Settings.json");
            configuration.Initialize();

            // Add WPFCommon
            builder.Services.AddWPFCommon<MainWindow, Settings>(configuration);

         //   builder.Services.AddSingleton<IMediaService, MediaService>();
            builder.Services.AddSingleton<ITextService, TextService>();

            _appHost = builder.Build();

            // Initialize WPFCommon
            _appHost.Services.UseWPFCommon();
        }


        /// <summary>
        /// Application startup.
        /// </summary>
        /// <returns>Task.</returns>
        private Task AppStartup()
        {
            MainWindow = _appHost.Services.GetMainWindow();
            MainWindow.Show();
            return Task.CompletedTask;
        }


        /// <summary>
        /// Application shutdown.
        /// </summary>
        private async Task AppShutdown()
        {
            using (_appHost)
            {
                await _appHost.StopAsync();
                DeregisterExceptionHandlers();
            }
        }


        /// <summary>
        /// Raises the <see cref="E:System.Windows.Application.Startup" /> event.
        /// </summary>
        /// <param name="e">A <see cref="T:System.Windows.StartupEventArgs" /> that contains the event data.</param>
        protected override async void OnStartup(StartupEventArgs e)
        {
            await AppStartup();
            base.OnStartup(e);
        }


        /// <summary>
        /// Raises the <see cref="E:System.Windows.Application.SessionEnding" /> event.
        /// </summary>
        /// <param name="e">A <see cref="T:System.Windows.SessionEndingCancelEventArgs" /> that contains the event data.</param>
        protected override async void OnSessionEnding(SessionEndingCancelEventArgs e)
        {
            await AppShutdown();
            base.OnSessionEnding(e);
        }


        /// <summary>
        /// Raises the <see cref="E:System.Windows.Application.Exit" /> event.
        /// </summary>
        /// <param name="e">An <see cref="T:System.Windows.ExitEventArgs" /> that contains the event data.</param>
        protected async override void OnExit(ExitEventArgs e)
        {
            await AppShutdown();
            base.OnExit(e);
        }


        /// <summary>
        /// Registers the exception handlers.
        /// </summary>
        private void RegisterExceptionHandlers()
        {
            DispatcherUnhandledException += OnDispatcherException;
            AppDomain.CurrentDomain.UnhandledException += OnAppDomainException;
            TaskScheduler.UnobservedTaskException += OnTaskSchedulerException;
        }


        /// <summary>
        /// Deregisters the exception handlers.
        /// </summary>
        private void DeregisterExceptionHandlers()
        {
            DispatcherUnhandledException -= OnDispatcherException;
            AppDomain.CurrentDomain.UnhandledException -= OnAppDomainException;
            TaskScheduler.UnobservedTaskException -= OnTaskSchedulerException;
        }


        /// <summary>
        /// Handles the <see cref="E:DispatcherException" /> event.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The <see cref="DispatcherUnhandledExceptionEventArgs"/> instance containing the event data.</param>
        private void OnDispatcherException(object sender, DispatcherUnhandledExceptionEventArgs e)
        {
            ShowExceptionMessage(e.Exception);

            // Prevent application from crashing
            e.Handled = true;
        }


        /// <summary>
        /// Handles the <see cref="E:AppDomainException" /> event.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The <see cref="UnhandledExceptionEventArgs"/> instance containing the event data.</param>
        private void OnAppDomainException(object sender, UnhandledExceptionEventArgs e)
        {
            if (e.ExceptionObject is Exception ex)
            {
                ShowExceptionMessage(ex);
            }
        }


        /// <summary>
        /// Handles the <see cref="E:TaskSchedulerException" /> event.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The <see cref="UnobservedTaskExceptionEventArgs"/> instance containing the event data.</param>
        private void OnTaskSchedulerException(object sender, UnobservedTaskExceptionEventArgs e)
        {
            ShowExceptionMessage(e.Exception);

            // Prevent application from crashing
            e.SetObserved();
        }


        private void ShowExceptionMessage(Exception ex)
        {
            MessageBox.Show($"An unexpected error occurred:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }
}


