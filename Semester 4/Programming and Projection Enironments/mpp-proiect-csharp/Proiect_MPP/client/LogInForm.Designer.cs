namespace client
{
    partial class LogInForm
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            welcomeLabel = new Label();
            entityCommand1 = new System.Data.Entity.Core.EntityClient.EntityCommand();
            passwordLabel = new Label();
            emailLabel = new Label();
            emailTextBox = new TextBox();
            passwordTextBox = new TextBox();
            logInButton = new Button();
            SuspendLayout();
            // 
            // welcomeLabel
            // 
            welcomeLabel.AutoSize = true;
            welcomeLabel.Font = new Font("Segoe UI", 20.25F, FontStyle.Regular, GraphicsUnit.Point);
            welcomeLabel.Location = new Point(86, 9);
            welcomeLabel.Name = "welcomeLabel";
            welcomeLabel.Size = new Size(127, 37);
            welcomeLabel.TabIndex = 0;
            welcomeLabel.Text = "Welcome";
            // 
            // entityCommand1
            // 
            entityCommand1.CommandTimeout = 0;
            entityCommand1.CommandTree = null;
            entityCommand1.Connection = null;
            entityCommand1.EnablePlanCaching = true;
            entityCommand1.Transaction = null;
            // 
            // passwordLabel
            // 
            passwordLabel.AutoSize = true;
            passwordLabel.Font = new Font("Segoe UI", 20.25F, FontStyle.Regular, GraphicsUnit.Point);
            passwordLabel.Location = new Point(12, 137);
            passwordLabel.Name = "passwordLabel";
            passwordLabel.Size = new Size(128, 37);
            passwordLabel.TabIndex = 1;
            passwordLabel.Text = "Password";
            // 
            // emailLabel
            // 
            emailLabel.AutoSize = true;
            emailLabel.Font = new Font("Segoe UI", 20.25F, FontStyle.Regular, GraphicsUnit.Point);
            emailLabel.Location = new Point(12, 71);
            emailLabel.Name = "emailLabel";
            emailLabel.Size = new Size(88, 37);
            emailLabel.TabIndex = 2;
            emailLabel.Text = "Email:";
            // 
            // emailTextBox
            // 
            emailTextBox.Location = new Point(12, 113);
            emailTextBox.Name = "emailTextBox";
            emailTextBox.Size = new Size(285, 23);
            emailTextBox.TabIndex = 3;
            // 
            // passwordTextBox
            // 
            passwordTextBox.Location = new Point(12, 177);
            passwordTextBox.Name = "passwordTextBox";
            passwordTextBox.PasswordChar = '*';
            passwordTextBox.Size = new Size(285, 23);
            passwordTextBox.TabIndex = 4;
            // 
            // logInButton
            // 
            logInButton.Font = new Font("Segoe UI", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            logInButton.Location = new Point(53, 223);
            logInButton.Name = "logInButton";
            logInButton.Size = new Size(208, 40);
            logInButton.TabIndex = 5;
            logInButton.Text = "Log In";
            logInButton.UseVisualStyleBackColor = true;
            logInButton.Click += logInButton_Click;
            // 
            // LogInForm
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(311, 295);
            Controls.Add(logInButton);
            Controls.Add(passwordTextBox);
            Controls.Add(emailTextBox);
            Controls.Add(emailLabel);
            Controls.Add(passwordLabel);
            Controls.Add(welcomeLabel);
            Name = "LogInForm";
            Text = "LogInForm";
            Load += LogInForm_Load;
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Label welcomeLabel;
        private System.Data.Entity.Core.EntityClient.EntityCommand entityCommand1;
        private Label passwordLabel;
        private Label emailLabel;
        private TextBox emailTextBox;
        private TextBox passwordTextBox;
        private Button logInButton;
    }
}