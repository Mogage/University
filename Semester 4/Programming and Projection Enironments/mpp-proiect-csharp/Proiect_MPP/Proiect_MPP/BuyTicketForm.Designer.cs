namespace Proiect_MPP
{
    partial class BuyTicketForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
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
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            buyTicketLabel = new Label();
            clientFirstNameLabel = new Label();
            buyTicketButton = new Button();
            clientLastNameLabel = new Label();
            clientAddressLabel = new Label();
            touristsLabel = new Label();
            clientFirstNameTextBox = new TextBox();
            clientLastNameTextBox = new TextBox();
            clientAddressTextBox = new TextBox();
            touristsTextBox = new TextBox();
            SuspendLayout();
            // 
            // buyTicketLabel
            // 
            buyTicketLabel.AutoSize = true;
            buyTicketLabel.Font = new Font("Monocraft", 15.7499981F, FontStyle.Bold, GraphicsUnit.Point);
            buyTicketLabel.Location = new Point(117, 9);
            buyTicketLabel.Name = "buyTicketLabel";
            buyTicketLabel.Size = new Size(160, 23);
            buyTicketLabel.TabIndex = 0;
            buyTicketLabel.Text = "Buy Ticket";
            // 
            // clientFirstNameLabel
            // 
            clientFirstNameLabel.AutoSize = true;
            clientFirstNameLabel.Font = new Font("Monocraft", 11.9999981F, FontStyle.Bold, GraphicsUnit.Point);
            clientFirstNameLabel.Location = new Point(12, 39);
            clientFirstNameLabel.Name = "clientFirstNameLabel";
            clientFirstNameLabel.Size = new Size(206, 18);
            clientFirstNameLabel.TabIndex = 1;
            clientFirstNameLabel.Text = "Client First Name:";
            // 
            // buyTicketButton
            // 
            buyTicketButton.Font = new Font("Monocraft", 11.25F, FontStyle.Bold, GraphicsUnit.Point);
            buyTicketButton.Location = new Point(117, 160);
            buyTicketButton.Name = "buyTicketButton";
            buyTicketButton.Size = new Size(160, 29);
            buyTicketButton.TabIndex = 3;
            buyTicketButton.Text = "Buy";
            buyTicketButton.UseVisualStyleBackColor = true;
            buyTicketButton.Click += buyTicketButton_Click;
            // 
            // clientLastNameLabel
            // 
            clientLastNameLabel.AutoSize = true;
            clientLastNameLabel.Font = new Font("Monocraft", 11.9999981F, FontStyle.Bold, GraphicsUnit.Point);
            clientLastNameLabel.Location = new Point(12, 68);
            clientLastNameLabel.Name = "clientLastNameLabel";
            clientLastNameLabel.Size = new Size(195, 18);
            clientLastNameLabel.TabIndex = 4;
            clientLastNameLabel.Text = "Client Last Name:";
            // 
            // clientAddressLabel
            // 
            clientAddressLabel.AutoSize = true;
            clientAddressLabel.Font = new Font("Monocraft", 11.9999981F, FontStyle.Bold, GraphicsUnit.Point);
            clientAddressLabel.Location = new Point(12, 97);
            clientAddressLabel.Name = "clientAddressLabel";
            clientAddressLabel.Size = new Size(173, 18);
            clientAddressLabel.TabIndex = 5;
            clientAddressLabel.Text = "Client Address:";
            // 
            // touristsLabel
            // 
            touristsLabel.AutoSize = true;
            touristsLabel.Font = new Font("Monocraft", 11.9999981F, FontStyle.Bold, GraphicsUnit.Point);
            touristsLabel.Location = new Point(12, 126);
            touristsLabel.Name = "touristsLabel";
            touristsLabel.Size = new Size(107, 18);
            touristsLabel.TabIndex = 6;
            touristsLabel.Text = "Tourists:";
            // 
            // clientFirstNameTextBox
            // 
            clientFirstNameTextBox.Location = new Point(224, 38);
            clientFirstNameTextBox.Name = "clientFirstNameTextBox";
            clientFirstNameTextBox.Size = new Size(173, 23);
            clientFirstNameTextBox.TabIndex = 7;
            // 
            // clientLastNameTextBox
            // 
            clientLastNameTextBox.Location = new Point(224, 67);
            clientLastNameTextBox.Name = "clientLastNameTextBox";
            clientLastNameTextBox.Size = new Size(173, 23);
            clientLastNameTextBox.TabIndex = 8;
            // 
            // clientAddressTextBox
            // 
            clientAddressTextBox.Location = new Point(224, 96);
            clientAddressTextBox.Name = "clientAddressTextBox";
            clientAddressTextBox.Size = new Size(173, 23);
            clientAddressTextBox.TabIndex = 9;
            // 
            // touristsTextBox
            // 
            touristsTextBox.Location = new Point(224, 125);
            touristsTextBox.Name = "touristsTextBox";
            touristsTextBox.Size = new Size(173, 23);
            touristsTextBox.TabIndex = 10;
            // 
            // BuyTicketForm
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(411, 201);
            Controls.Add(touristsTextBox);
            Controls.Add(clientAddressTextBox);
            Controls.Add(clientLastNameTextBox);
            Controls.Add(clientFirstNameTextBox);
            Controls.Add(touristsLabel);
            Controls.Add(clientAddressLabel);
            Controls.Add(clientLastNameLabel);
            Controls.Add(buyTicketButton);
            Controls.Add(clientFirstNameLabel);
            Controls.Add(buyTicketLabel);
            Name = "BuyTicketForm";
            Text = "BuyTicketForm";
            Load += BuyTicketForm_Load;
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Label buyTicketLabel;
        private Label clientFirstNameLabel;
        private Button buyTicketButton;
        private Label clientLastNameLabel;
        private Label clientAddressLabel;
        private Label touristsLabel;
        private TextBox clientFirstNameTextBox;
        private TextBox clientLastNameTextBox;
        private TextBox clientAddressTextBox;
        private TextBox touristsTextBox;
    }
}