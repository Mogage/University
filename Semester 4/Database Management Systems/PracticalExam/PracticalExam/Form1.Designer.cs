namespace PracticalExam
{
    partial class Form1
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
            dataGridViewParent = new DataGridView();
            sqlCommand1 = new Microsoft.Data.SqlClient.SqlCommand();
            dataGridViewChild = new DataGridView();
            buttonAdd = new Button();
            buttonDelete = new Button();
            buttonUpdate = new Button();
            ((System.ComponentModel.ISupportInitialize)dataGridViewParent).BeginInit();
            ((System.ComponentModel.ISupportInitialize)dataGridViewChild).BeginInit();
            SuspendLayout();
            // 
            // dataGridViewParent
            // 
            dataGridViewParent.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            dataGridViewParent.Location = new Point(12, 12);
            dataGridViewParent.Name = "dataGridViewParent";
            dataGridViewParent.RowTemplate.Height = 25;
            dataGridViewParent.Size = new Size(558, 164);
            dataGridViewParent.TabIndex = 0;
            // 
            // sqlCommand1
            // 
            sqlCommand1.CommandTimeout = 30;
            sqlCommand1.EnableOptimizedParameterBinding = false;
            // 
            // dataGridViewChild
            // 
            dataGridViewChild.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            dataGridViewChild.Location = new Point(12, 192);
            dataGridViewChild.Name = "dataGridViewChild";
            dataGridViewChild.RowTemplate.Height = 25;
            dataGridViewChild.Size = new Size(785, 199);
            dataGridViewChild.TabIndex = 1;
            // 
            // buttonAdd
            // 
            buttonAdd.Location = new Point(639, 12);
            buttonAdd.Name = "buttonAdd";
            buttonAdd.Size = new Size(121, 23);
            buttonAdd.TabIndex = 2;
            buttonAdd.Text = "Add Profile";
            buttonAdd.UseVisualStyleBackColor = true;
            buttonAdd.Click += buttonAdd_Click;
            // 
            // buttonDelete
            // 
            buttonDelete.Location = new Point(639, 153);
            buttonDelete.Name = "buttonDelete";
            buttonDelete.Size = new Size(121, 23);
            buttonDelete.TabIndex = 3;
            buttonDelete.Text = "Delete Profile";
            buttonDelete.UseVisualStyleBackColor = true;
            buttonDelete.Click += buttonDelete_Click;
            // 
            // buttonUpdate
            // 
            buttonUpdate.Location = new Point(639, 79);
            buttonUpdate.Name = "buttonUpdate";
            buttonUpdate.Size = new Size(121, 23);
            buttonUpdate.TabIndex = 4;
            buttonUpdate.Text = "Update Profile";
            buttonUpdate.UseVisualStyleBackColor = true;
            buttonUpdate.Click += buttonUpdate_Click;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(837, 403);
            Controls.Add(buttonUpdate);
            Controls.Add(buttonDelete);
            Controls.Add(buttonAdd);
            Controls.Add(dataGridViewChild);
            Controls.Add(dataGridViewParent);
            Name = "Form1";
            Text = "Form1";
            Load += Form1_Load;
            ((System.ComponentModel.ISupportInitialize)dataGridViewParent).EndInit();
            ((System.ComponentModel.ISupportInitialize)dataGridViewChild).EndInit();
            ResumeLayout(false);
        }

        #endregion

        private DataGridView dataGridViewParent;
        private Microsoft.Data.SqlClient.SqlCommand sqlCommand1;
        private DataGridView dataGridViewChild;
        private Button buttonAdd;
        private Button buttonDelete;
        private Button buttonUpdate;
    }
}