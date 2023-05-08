namespace client
{
    partial class MainForm
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
            allFlightsTable = new DataGridView();
            idColumn = new DataGridViewTextBoxColumn();
            departureCityColumn = new DataGridViewTextBoxColumn();
            departureColumn = new DataGridViewTextBoxColumn();
            destinationCityColumn = new DataGridViewTextBoxColumn();
            destinationName = new DataGridViewTextBoxColumn();
            dateColumn = new DataGridViewTextBoxColumn();
            timeColumn = new DataGridViewTextBoxColumn();
            freeSeatsColumn = new DataGridViewTextBoxColumn();
            label1 = new Label();
            destinationTextBox = new TextBox();
            datePicker = new DateTimePicker();
            buyTicketsButton = new Button();
            logOutButton = new Button();
            searchFlightsTable = new DataGridView();
            searchIdColumn = new DataGridViewTextBoxColumn();
            searchDepartureCityColumn = new DataGridViewTextBoxColumn();
            searchDepartureColumn = new DataGridViewTextBoxColumn();
            searchDestinationCityColumn = new DataGridViewTextBoxColumn();
            searchDestinationName = new DataGridViewTextBoxColumn();
            searchDateColumn = new DataGridViewTextBoxColumn();
            searchTimeColumn = new DataGridViewTextBoxColumn();
            searchFreeSeatsColumn = new DataGridViewTextBoxColumn();
            searchButton = new Button();
            ((System.ComponentModel.ISupportInitialize)allFlightsTable).BeginInit();
            ((System.ComponentModel.ISupportInitialize)searchFlightsTable).BeginInit();
            SuspendLayout();
            // 
            // allFlightsTable
            // 
            allFlightsTable.AllowUserToAddRows = false;
            allFlightsTable.AllowUserToDeleteRows = false;
            allFlightsTable.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            allFlightsTable.Columns.AddRange(new DataGridViewColumn[] { idColumn, departureCityColumn, departureColumn, destinationCityColumn, destinationName, dateColumn, timeColumn, freeSeatsColumn });
            allFlightsTable.Location = new Point(12, 12);
            allFlightsTable.Name = "allFlightsTable";
            allFlightsTable.RowTemplate.Height = 25;
            allFlightsTable.Size = new Size(760, 195);
            allFlightsTable.TabIndex = 0;
            // 
            // idColumn
            // 
            idColumn.HeaderText = "idColumn";
            idColumn.Name = "idColumn";
            idColumn.Visible = false;
            // 
            // departureCityColumn
            // 
            departureCityColumn.FillWeight = 115F;
            departureCityColumn.HeaderText = "Departure City";
            departureCityColumn.Name = "departureCityColumn";
            departureCityColumn.ReadOnly = true;
            departureCityColumn.Resizable = DataGridViewTriState.False;
            departureCityColumn.Width = 117;
            // 
            // departureColumn
            // 
            departureColumn.AutoSizeMode = DataGridViewAutoSizeColumnMode.None;
            departureColumn.HeaderText = "Airport Name";
            departureColumn.Name = "departureColumn";
            departureColumn.ReadOnly = true;
            departureColumn.Resizable = DataGridViewTriState.False;
            // 
            // destinationCityColumn
            // 
            destinationCityColumn.HeaderText = "Destination City";
            destinationCityColumn.Name = "destinationCityColumn";
            destinationCityColumn.ReadOnly = true;
            destinationCityColumn.Resizable = DataGridViewTriState.False;
            // 
            // destinationName
            // 
            destinationName.HeaderText = "Airport Name";
            destinationName.Name = "destinationName";
            destinationName.ReadOnly = true;
            destinationName.Resizable = DataGridViewTriState.False;
            // 
            // dateColumn
            // 
            dateColumn.HeaderText = "Departure Date";
            dateColumn.Name = "dateColumn";
            dateColumn.ReadOnly = true;
            dateColumn.Resizable = DataGridViewTriState.False;
            // 
            // timeColumn
            // 
            timeColumn.HeaderText = "Departure Time";
            timeColumn.Name = "timeColumn";
            timeColumn.ReadOnly = true;
            timeColumn.Resizable = DataGridViewTriState.False;
            // 
            // freeSeatsColumn
            // 
            freeSeatsColumn.HeaderText = "Free Seats";
            freeSeatsColumn.Name = "freeSeatsColumn";
            freeSeatsColumn.ReadOnly = true;
            freeSeatsColumn.Resizable = DataGridViewTriState.False;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Font = new Font("Monocraft", 15.7499981F, FontStyle.Bold, GraphicsUnit.Point);
            label1.Location = new Point(14, 218);
            label1.Name = "label1";
            label1.Size = new Size(115, 23);
            label1.TabIndex = 2;
            label1.Text = "Search:";
            // 
            // destinationTextBox
            // 
            destinationTextBox.Font = new Font("Monocraft", 9.749999F, FontStyle.Bold, GraphicsUnit.Point);
            destinationTextBox.Location = new Point(125, 218);
            destinationTextBox.Name = "destinationTextBox";
            destinationTextBox.PlaceholderText = "Destination";
            destinationTextBox.Size = new Size(134, 22);
            destinationTextBox.TabIndex = 3;
            // 
            // datePicker
            // 
            datePicker.CalendarFont = new Font("Monocraft", 9.749999F, FontStyle.Bold, GraphicsUnit.Point);
            datePicker.Font = new Font("Segoe UI", 9F, FontStyle.Bold, GraphicsUnit.Point);
            datePicker.Location = new Point(265, 218);
            datePicker.Name = "datePicker";
            datePicker.Size = new Size(204, 23);
            datePicker.TabIndex = 4;
            // 
            // buyTicketsButton
            // 
            buyTicketsButton.Font = new Font("Monocraft", 9.749999F, FontStyle.Bold, GraphicsUnit.Point);
            buyTicketsButton.Location = new Point(573, 218);
            buyTicketsButton.Name = "buyTicketsButton";
            buyTicketsButton.Size = new Size(114, 25);
            buyTicketsButton.TabIndex = 5;
            buyTicketsButton.Text = "Buy Tickets";
            buyTicketsButton.UseVisualStyleBackColor = true;
            buyTicketsButton.Click += buyTicketsButton_Click;
            // 
            // logOutButton
            // 
            logOutButton.Font = new Font("Monocraft", 9.749999F, FontStyle.Bold, GraphicsUnit.Point);
            logOutButton.Location = new Point(693, 218);
            logOutButton.Name = "logOutButton";
            logOutButton.Size = new Size(79, 25);
            logOutButton.TabIndex = 6;
            logOutButton.Text = "Log out";
            logOutButton.UseVisualStyleBackColor = true;
            logOutButton.Click += logOutButton_Click;
            // 
            // searchFlightsTable
            // 
            searchFlightsTable.AllowUserToAddRows = false;
            searchFlightsTable.AllowUserToDeleteRows = false;
            searchFlightsTable.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            searchFlightsTable.Columns.AddRange(new DataGridViewColumn[] { searchIdColumn, searchDepartureCityColumn, searchDepartureColumn, searchDestinationCityColumn, searchDestinationName, searchDateColumn, searchTimeColumn, searchFreeSeatsColumn });
            searchFlightsTable.Location = new Point(12, 254);
            searchFlightsTable.Name = "searchFlightsTable";
            searchFlightsTable.RowTemplate.Height = 25;
            searchFlightsTable.Size = new Size(760, 195);
            searchFlightsTable.TabIndex = 7;
            searchFlightsTable.CellContentClick += searchFlightsTable_CellContentClick;
            // 
            // searchIdColumn
            // 
            searchIdColumn.HeaderText = "searchIdColumn";
            searchIdColumn.Name = "searchIdColumn";
            searchIdColumn.Visible = false;
            // 
            // searchDepartureCityColumn
            // 
            searchDepartureCityColumn.FillWeight = 117F;
            searchDepartureCityColumn.HeaderText = "Departure City";
            searchDepartureCityColumn.Name = "searchDepartureCityColumn";
            searchDepartureCityColumn.ReadOnly = true;
            searchDepartureCityColumn.Resizable = DataGridViewTriState.False;
            searchDepartureCityColumn.Width = 117;
            // 
            // searchDepartureColumn
            // 
            searchDepartureColumn.AutoSizeMode = DataGridViewAutoSizeColumnMode.None;
            searchDepartureColumn.HeaderText = "Airport Name";
            searchDepartureColumn.Name = "searchDepartureColumn";
            searchDepartureColumn.ReadOnly = true;
            searchDepartureColumn.Resizable = DataGridViewTriState.False;
            // 
            // searchDestinationCityColumn
            // 
            searchDestinationCityColumn.HeaderText = "Destination City";
            searchDestinationCityColumn.Name = "searchDestinationCityColumn";
            searchDestinationCityColumn.ReadOnly = true;
            searchDestinationCityColumn.Resizable = DataGridViewTriState.False;
            // 
            // searchDestinationName
            // 
            searchDestinationName.HeaderText = "Airport Name";
            searchDestinationName.Name = "searchDestinationName";
            searchDestinationName.ReadOnly = true;
            searchDestinationName.Resizable = DataGridViewTriState.False;
            // 
            // searchDateColumn
            // 
            searchDateColumn.HeaderText = "Departure Date";
            searchDateColumn.Name = "searchDateColumn";
            searchDateColumn.ReadOnly = true;
            searchDateColumn.Resizable = DataGridViewTriState.False;
            // 
            // searchTimeColumn
            // 
            searchTimeColumn.HeaderText = "Departure Time";
            searchTimeColumn.Name = "searchTimeColumn";
            searchTimeColumn.ReadOnly = true;
            searchTimeColumn.Resizable = DataGridViewTriState.False;
            // 
            // searchFreeSeatsColumn
            // 
            searchFreeSeatsColumn.HeaderText = "Free Seats";
            searchFreeSeatsColumn.Name = "searchFreeSeatsColumn";
            searchFreeSeatsColumn.ReadOnly = true;
            searchFreeSeatsColumn.Resizable = DataGridViewTriState.False;
            // 
            // searchButton
            // 
            searchButton.Font = new Font("Monocraft", 9.749999F, FontStyle.Bold, GraphicsUnit.Point);
            searchButton.Location = new Point(475, 218);
            searchButton.Name = "searchButton";
            searchButton.Size = new Size(92, 25);
            searchButton.TabIndex = 8;
            searchButton.Text = "Search";
            searchButton.UseVisualStyleBackColor = true;
            searchButton.Click += searchButton_Click;
            // 
            // MainForm
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(784, 461);
            Controls.Add(searchButton);
            Controls.Add(searchFlightsTable);
            Controls.Add(logOutButton);
            Controls.Add(buyTicketsButton);
            Controls.Add(datePicker);
            Controls.Add(destinationTextBox);
            Controls.Add(label1);
            Controls.Add(allFlightsTable);
            Name = "MainForm";
            Text = "MainForm";
            Load += MainForm_Load;
            ((System.ComponentModel.ISupportInitialize)allFlightsTable).EndInit();
            ((System.ComponentModel.ISupportInitialize)searchFlightsTable).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private DataGridView allFlightsTable;
        private Label label1;
        private TextBox destinationTextBox;
        private DateTimePicker datePicker;
        private Button buyTicketsButton;
        private Button logOutButton;
        private DataGridView searchFlightsTable;
        private DataGridViewTextBoxColumn idColumn;
        private DataGridViewTextBoxColumn departureCityColumn;
        private DataGridViewTextBoxColumn departureColumn;
        private DataGridViewTextBoxColumn destinationCityColumn;
        private DataGridViewTextBoxColumn destinationName;
        private DataGridViewTextBoxColumn dateColumn;
        private DataGridViewTextBoxColumn timeColumn;
        private DataGridViewTextBoxColumn freeSeatsColumn;
        private DataGridViewTextBoxColumn searchIdColumn;
        private DataGridViewTextBoxColumn searchDepartureCityColumn;
        private DataGridViewTextBoxColumn searchDepartureColumn;
        private DataGridViewTextBoxColumn searchDestinationCityColumn;
        private DataGridViewTextBoxColumn searchDestinationName;
        private DataGridViewTextBoxColumn searchDateColumn;
        private DataGridViewTextBoxColumn searchTimeColumn;
        private DataGridViewTextBoxColumn searchFreeSeatsColumn;
        private Button searchButton;
    }
}