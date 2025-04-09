from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from interview_corvus.config import PromptTemplates, settings
from interview_corvus.core.prompt_manager import PromptManager
from interview_corvus.security.api_key_manager import APIKeyManager
from interview_corvus.ui.hotkey_edit import HotkeyEdit


class SettingsDialog(QDialog):
    """
    Dialog for configuring application settings.
    """

    def __init__(self, parent=None):
        """
        Initialize the settings dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.api_key_manager = APIKeyManager()
        self.prompt_manager = PromptManager()

        self.setWindowTitle("Settings")
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Create a tab widget for better organization
        self.tab_widget = QTabWidget()

        # General tab
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        # Language selection
        language_group = QGroupBox("Programming Language")
        language_layout = QFormLayout()

        self.language_combo = QComboBox()
        self.language_combo.addItems(settings.available_languages)
        language_layout.addRow("Default Language:", self.language_combo)

        language_group.setLayout(language_layout)
        general_layout.addWidget(language_group)

        # Add the general tab
        self.tab_widget.addTab(general_tab, "General")

        # LLM Settings tab
        llm_tab = QWidget()
        llm_layout = QVBoxLayout(llm_tab)

        # LLM Settings
        llm_group = QGroupBox("LLM Settings")
        llm_form_layout = QFormLayout()

        # API Provider
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["OpenAI", "Anthropic"])
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        llm_form_layout.addRow("API Provider:", self.provider_combo)

        # Model selection
        self.model_combo = QComboBox()
        # Add all available models
        for model_id, model_name in settings.llm.available_models.items():
            self.model_combo.addItem(model_name, model_id)
        llm_form_layout.addRow("Model:", self.model_combo)

        # OpenRouter settings
        self.use_openrouter = QCheckBox("Use OpenRouter")
        self.use_openrouter.setChecked(settings.llm.use_openrouter)
        llm_form_layout.addRow("", self.use_openrouter)

        # API Key
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Your API key")
        llm_form_layout.addRow("API Key:", self.api_key_input)

        # Show API Key checkbox
        self.show_api_key = QCheckBox("Show API Key")
        self.show_api_key.stateChanged.connect(self.toggle_api_key_visibility)
        llm_form_layout.addRow("", self.show_api_key)

        # Temperature
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(0.0, 2.0)
        self.temperature_input.setSingleStep(0.1)
        self.temperature_input.setDecimals(1)
        self.temperature_input.setValue(1.0)
        llm_form_layout.addRow("Temperature:", self.temperature_input)

        # Test connection button
        self.test_connection_button = QPushButton("Check connection")
        self.test_connection_button.clicked.connect(self.test_connection)
        llm_form_layout.addRow("", self.test_connection_button)
        
        # Clear Settings Cache button
        self.clear_cache_button = QPushButton("Clear Settings Cache")
        self.clear_cache_button.clicked.connect(self.clear_settings_cache)
        llm_form_layout.addRow("", self.clear_cache_button)

        llm_group.setLayout(llm_form_layout)
        llm_layout.addWidget(llm_group)

        # Add the LLM tab
        self.tab_widget.addTab(llm_tab, "LLM Settings")

        # Prompts tab
        prompts_tab = QWidget()
        prompts_layout = QVBoxLayout(prompts_tab)

        # Prompt editor
        prompt_group = QGroupBox("Prompt Templates")
        prompt_layout = QVBoxLayout()

        # List of available prompts
        prompt_select_layout = QHBoxLayout()
        prompt_select_layout.addWidget(QLabel("Available Templates:"))

        self.prompt_list = QListWidget()
        self.prompt_list.addItems(self.prompt_manager.get_all_template_names())
        self.prompt_list.currentTextChanged.connect(self.on_prompt_selected)
        prompt_select_layout.addWidget(self.prompt_list)
        prompt_layout.addLayout(prompt_select_layout)

        # Template editor
        self.prompt_editor = QTextEdit()
        self.prompt_editor.setPlaceholderText("Select a template to edit")
        prompt_layout.addWidget(self.prompt_editor)

        # Save template button
        self.save_prompt_button = QPushButton("Save Template")
        self.save_prompt_button.clicked.connect(self.save_prompt_template)
        prompt_layout.addWidget(self.save_prompt_button)

        # Reset to default button
        self.reset_prompt_button = QPushButton("Reset to Default")
        self.reset_prompt_button.clicked.connect(self.reset_prompt_template)
        prompt_layout.addWidget(self.reset_prompt_button)

        prompt_group.setLayout(prompt_layout)
        prompts_layout.addWidget(prompt_group)

        # Add the prompts tab
        self.tab_widget.addTab(prompts_tab, "Prompts")

        # Add tab widget to main layout
        layout.addWidget(self.tab_widget)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.save_settings)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        hotkeys_tab = QWidget()
        hotkeys_layout = QVBoxLayout(hotkeys_tab)

        # Group for main hotkeys
        main_hotkeys_group = QGroupBox("Main Hotkeys")
        main_hotkeys_layout = QFormLayout()

        # Screenshot hotkey
        self.screenshot_hotkey = HotkeyEdit(settings.hotkeys.screenshot_key)
        main_hotkeys_layout.addRow("Take Screenshot:", self.screenshot_hotkey)

        # Generate solution hotkey
        self.generate_solution_hotkey = HotkeyEdit(
            settings.hotkeys.generate_solution_key
        )
        main_hotkeys_layout.addRow("Generate Solution:", self.generate_solution_hotkey)

        # Toggle visibility hotkey
        self.toggle_visibility_hotkey = HotkeyEdit(
            settings.hotkeys.toggle_visibility_key
        )
        main_hotkeys_layout.addRow("Toggle Visibility:", self.toggle_visibility_hotkey)

        # Optimize solution hotkey
        self.optimize_solution_hotkey = HotkeyEdit(
            settings.hotkeys.optimize_solution_key
        )
        main_hotkeys_layout.addRow("Optimize Solution:", self.optimize_solution_hotkey)

        # Reset history hotkey
        self.reset_history_hotkey = HotkeyEdit(settings.hotkeys.reset_history_key)
        main_hotkeys_layout.addRow("Reset History:", self.reset_history_hotkey)

        # Panic hotkey
        self.panic_hotkey = HotkeyEdit(settings.hotkeys.panic_key)
        main_hotkeys_layout.addRow("Panic (Instant Hide):", self.panic_hotkey)

        main_hotkeys_group.setLayout(main_hotkeys_layout)
        hotkeys_layout.addWidget(main_hotkeys_group)

        # Group for window movement hotkeys
        move_hotkeys_group = QGroupBox("Window Movement Hotkeys")
        move_hotkeys_layout = QFormLayout()

        # Move window up hotkey
        self.move_up_hotkey = HotkeyEdit(settings.hotkeys.move_window_keys["up"])
        move_hotkeys_layout.addRow("Move Window Up:", self.move_up_hotkey)

        # Move window down hotkey
        self.move_down_hotkey = HotkeyEdit(settings.hotkeys.move_window_keys["down"])
        move_hotkeys_layout.addRow("Move Window Down:", self.move_down_hotkey)

        # Move window left hotkey
        self.move_left_hotkey = HotkeyEdit(settings.hotkeys.move_window_keys["left"])
        move_hotkeys_layout.addRow("Move Window Left:", self.move_left_hotkey)

        # Move window right hotkey
        self.move_right_hotkey = HotkeyEdit(settings.hotkeys.move_window_keys["right"])
        move_hotkeys_layout.addRow("Move Window Right:", self.move_right_hotkey)

        move_hotkeys_group.setLayout(move_hotkeys_layout)
        hotkeys_layout.addWidget(move_hotkeys_group)

        # Help text
        help_label = QLabel(
            "Click in the input field and press the desired key combination.\n"
            "All hotkeys require at least one modifier key (Ctrl, Alt, Shift, Cmd/Win).\n"
            "Changes will take effect after saving settings."
        )
        help_label.setWordWrap(True)
        hotkeys_layout.addWidget(help_label)

        # Reset button
        reset_hotkeys_button = QPushButton("Reset All Hotkeys to Defaults")
        reset_hotkeys_button.clicked.connect(self.reset_hotkeys)
        hotkeys_layout.addWidget(reset_hotkeys_button)

        # Add spacer
        hotkeys_layout.addStretch()

        # Add the hotkeys tab
        self.tab_widget.addTab(hotkeys_tab, "Hotkeys")

    def load_settings(self):
        """Load current settings into the UI."""
        # Default programming language
        index = self.language_combo.findText(settings.default_language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)

        # Default to OpenAI
        self.provider_combo.setCurrentText("OpenAI")

        # LLM settings
        index = self.model_combo.findData(settings.llm.model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.temperature_input.setValue(settings.llm.temperature)
        self.use_openrouter.setChecked(settings.llm.use_openrouter)

        # Get API key if it exists
        try:
            api_key = self.api_key_manager.get_api_key()
            if api_key:
                self.api_key_input.setText(api_key)
        except (ValueError, Exception) as e:
            logger.info(f"Could not load API key: {e}")
            # No API key exists

    def save_settings(self):
        """Save user-specific settings to a JSON file."""
        # Save language setting
        settings.default_language = self.language_combo.currentText()

        # Save LLM settings
        settings.llm.model = self.model_combo.currentData()
        settings.llm.temperature = self.temperature_input.value()
        settings.llm.use_openrouter = self.use_openrouter.isChecked()

        if self.screenshot_hotkey.text():
            settings.hotkeys.screenshot_key = self.screenshot_hotkey.text()
        if self.generate_solution_hotkey.text():
            settings.hotkeys.generate_solution_key = (
                self.generate_solution_hotkey.text()
            )
        if self.toggle_visibility_hotkey.text():
            settings.hotkeys.toggle_visibility_key = (
                self.toggle_visibility_hotkey.text()
            )
        if self.optimize_solution_hotkey.text():
            settings.hotkeys.optimize_solution_key = (
                self.optimize_solution_hotkey.text()
            )
        if self.reset_history_hotkey.text():
            settings.hotkeys.reset_history_key = self.reset_history_hotkey.text()
        if self.panic_hotkey.text():
            settings.hotkeys.panic_key = self.panic_hotkey.text()

            # Save move window hotkeys
        move_window_keys = settings.hotkeys.move_window_keys.copy()
        if self.move_up_hotkey.text():
            move_window_keys["up"] = self.move_up_hotkey.text()
        if self.move_down_hotkey.text():
            move_window_keys["down"] = self.move_down_hotkey.text()
        if self.move_left_hotkey.text():
            move_window_keys["left"] = self.move_left_hotkey.text()
        if self.move_right_hotkey.text():
            move_window_keys["right"] = self.move_right_hotkey.text()
        settings.hotkeys.move_window_keys = move_window_keys

        # Save API key
        api_key = self.api_key_input.text().strip()
        if api_key:
            try:
                self.api_key_manager.set_api_key(api_key)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error saving API key: {e}")

        try:
            settings.save_user_settings()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error saving settings: {e}")

        # Close dialog
        self.accept()

    def on_provider_changed(self, provider):
        """
        Handle provider change.

        Args:
            provider: The selected provider name
        """
        # Save current model data and selection
        current_model_data = None
        if self.model_combo.currentIndex() >= 0:
            current_model_data = self.model_combo.currentData()
        
        # Clear the model combo
        self.model_combo.clear()

        if provider == "OpenAI":
            # Add standard OpenAI models
            standard_models = [
                ("gpt-4o", "OpenAI GPT-4"),
                ("o3-mini", "OpenAI GPT-3.5 Mini"),
                ("gpt-4o-mini", "OpenAI GPT-4 Mini"),
                ("o1", "OpenAI GPT-3.5"),
                ("o1-mini", "OpenAI GPT-3.5 Mini"),
            ]
            
            # Also add OpenRouter models
            openrouter_models = {k: v for k, v in settings.llm.available_models.items() 
                               if "deepseek" in k or "nvidia" in k}
            
            # Add standard OpenAI models
            for model_id, model_name in standard_models:
                self.model_combo.addItem(model_name, model_id)
                
            # Add OpenRouter models if any
            if openrouter_models:
                for model_id, model_name in openrouter_models.items():
                    self.model_combo.addItem(model_name, model_id)
                    
            self.api_key_input.setPlaceholderText("Enter OpenAI or OpenRouter API key")
            
        elif provider == "Anthropic":
            self.model_combo.addItems(
                ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"]
            )
            self.api_key_input.setPlaceholderText("Enter Anthropic API key")

        # Try to restore previous selection or set default
        if current_model_data and provider == "OpenAI":
            index = self.model_combo.findData(current_model_data)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                return
                
        # Set default model based on provider
        if provider == "OpenAI":
            self.model_combo.setCurrentText("OpenAI GPT-4")  # Set default
        else:
            self.model_combo.setCurrentText("claude-3-7-sonnet-latest")

    def toggle_api_key_visibility(self, state):
        """
        Toggle API key visibility.

        Args:
            state: Checkbox state
        """
        if state == Qt.CheckState.Checked.value:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)

    def test_connection(self):
        """Test connection to the LLM API."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Error", "Please enter an API key.")
            return

        # Determine if we're using OpenRouter based on either checkbox or model name
        model_id = self.model_combo.currentData()
        use_openrouter = self.use_openrouter.isChecked() or any(prefix in model_id for prefix in ["deepseek", "nvidia"])
        
        # Temporarily save API key for testing
        try:
            old_api_key = self.api_key_manager.get_api_key()
            self.api_key_manager.set_api_key(api_key)

            self.test_connection_button.setEnabled(False)
            self.test_connection_button.setText("Checking connection...")
            QApplication.processEvents()  # Update UI

            # Import appropriate library based on provider
            if use_openrouter:
                # Use direct requests for OpenRouter to ensure proper authentication
                import requests
                import json
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/afaneor/interview-corvus",
                    "X-Title": "Interview Corvus"
                }
                
                data = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5  # Minimal tokens for testing
                }
                
                logger.info(f"Testing OpenRouter connection with model: {model_id}")
                
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data  # Use json parameter instead of data with json.dumps
                )
                
                # Print response for debugging
                logger.info(f"OpenRouter response status: {response.status_code}")
                if response.status_code != 200:
                    response_text = response.text
                    logger.error(f"OpenRouter error: {response_text}")
                    raise Exception(f"Error code: {response.status_code} - {response_text}")
                
            elif self.provider_combo.currentText() == "OpenAI":
                from openai import OpenAI

                client = OpenAI(api_key=api_key)
                client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "Hello, are you working?"}],
                )
            else:  # Anthropic
                from anthropic import Anthropic

                client = Anthropic(api_key=api_key)
                client.messages.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "Hello, are you working?"}],
                )

            if old_api_key:
                self.api_key_manager.set_api_key(old_api_key)

            QMessageBox.information(self, "Success", "Connection successful!")

        except Exception as e:
            if old_api_key:
                self.api_key_manager.set_api_key(old_api_key)

            QMessageBox.critical(self, "Error", f"Cannot connect to API: {str(e)}")
        finally:
            self.test_connection_button.setEnabled(True)
            self.test_connection_button.setText("Check connection")

    def on_prompt_selected(self, template_name):
        """
        Handle prompt template selection.

        Args:
            template_name: The name of the selected template
        """
        if not template_name:
            self.prompt_editor.clear()
            self.prompt_editor.setPlaceholderText("Select a template to edit")
            return

        try:
            template_text = self.prompt_manager.get_template(template_name)
            self.prompt_editor.setText(template_text)
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def save_prompt_template(self):
        """Save the current prompt template."""
        current_template = self.prompt_list.currentItem()
        if not current_template:
            QMessageBox.warning(self, "Error", "Please select a template to save")
            return

        template_name = current_template.text()
        template_text = self.prompt_editor.toPlainText()

        if not template_text.strip():
            QMessageBox.warning(self, "Error", "Template cannot be empty")
            return

        try:
            self.prompt_manager.update_template(template_name, template_text)
            QMessageBox.information(
                self, "Success", f"Template '{template_name}' saved successfully"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving template: {str(e)}")

    def reset_prompt_template(self):
        """Reset the current prompt template to default."""
        current_template = self.prompt_list.currentItem()
        if not current_template:
            QMessageBox.warning(self, "Error", "Please select a template to reset")
            return

        template_name = current_template.text()

        # Confirm reset
        reply = QMessageBox.question(
            self,
            "Confirm Reset",
            f"Are you sure you want to reset the '{template_name}' template to default?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # Get default template from initial settings
        default_templates = PromptTemplates().templates
        if template_name in default_templates:
            default_template = default_templates[template_name]
            self.prompt_editor.setText(default_template)
            self.prompt_manager.update_template(template_name, default_template)
            QMessageBox.information(
                self, "Success", f"Template '{template_name}' reset to default"
            )
        else:
            QMessageBox.warning(
                self, "Error", f"No default template found for '{template_name}'"
            )

    def reset_hotkeys(self):
        """Reset hotkeys to their default values."""
        # Reset to defaults
        settings.hotkeys.reset_to_defaults()

        # Update UI with defaults
        self.screenshot_hotkey.setText(settings.hotkeys.screenshot_key)
        self.generate_solution_hotkey.setText(settings.hotkeys.generate_solution_key)
        self.toggle_visibility_hotkey.setText(settings.hotkeys.toggle_visibility_key)
        self.optimize_solution_hotkey.setText(settings.hotkeys.optimize_solution_key)
        self.reset_history_hotkey.setText(settings.hotkeys.reset_history_key)
        self.panic_hotkey.setText(settings.hotkeys.panic_key)

        self.move_up_hotkey.setText(settings.hotkeys.move_window_keys["up"])
        self.move_down_hotkey.setText(settings.hotkeys.move_window_keys["down"])
        self.move_left_hotkey.setText(settings.hotkeys.move_window_keys["left"])
        self.move_right_hotkey.setText(settings.hotkeys.move_window_keys["right"])

        QMessageBox.information(
            self, "Reset Hotkeys", "Hotkeys have been reset to default values."
        )

    def clear_settings_cache(self):
        """Clear the settings cache."""
        try:
            # Get the settings file path
            settings_path = settings.app_data_dir / "user_settings.json"
            
            # Delete the file if it exists
            if settings_path.exists():
                settings_path.unlink()
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Settings cache cleared successfully.\nDeleted: {settings_path}\n\nPlease restart the application."
                )
            else:
                QMessageBox.information(
                    self, 
                    "Info", 
                    f"No settings cache found at {settings_path}"
                )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to clear settings cache: {str(e)}"
            )
