using System.ComponentModel.Design;
using System.Data;
using Microsoft.Data.SqlClient;

namespace TermProject
{
    public partial class Form1 : Form
    {
        string connectionString = @"Server=DESKTOP-P6FMEQ4;Database=Laborator_01;Integrated Security=true;TrustServerCertificate=true;";
        DataSet ds = new DataSet();
        SqlDataAdapter parentAdapter = new SqlDataAdapter();
        SqlDataAdapter childAdapter = new SqlDataAdapter();
        BindingSource parentBS = new BindingSource();
        BindingSource childBS = new BindingSource();

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {
                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();
                    parentAdapter.SelectCommand = new SqlCommand("SELECT * FROM Users", connection);
                    childAdapter.SelectCommand = new SqlCommand("SELECT * FROM Profiles", connection);
                    parentAdapter.Fill(ds, "Users");
                    childAdapter.Fill(ds, "Profiles");
                    parentBS.DataSource = ds.Tables["Users"];
                    dataGridViewParent.DataSource = parentBS;
                    //dataGridViewParent.Columns[0].Visible = false;
                    //dataGridViewParent.Columns[1].Width = 175;
                    //dataGridViewParent.Columns[2].Visible = false;
                    DataColumn parentColumn = ds.Tables["Users"].Columns["idUser"];
                    DataColumn childColumn = ds.Tables["Profiles"].Columns["idUser"];
                    DataRelation relation = new DataRelation("FK_Users_Profiles", parentColumn, childColumn);
                    ds.Relations.Add(relation);
                    childBS.DataSource = parentBS;
                    childBS.DataMember = "FK_Users_Profiles";
                    dataGridViewChild.DataSource = childBS;
                    //dataGridViewChild.Columns[0].Visible = false;
                    //dataGridViewChild.Columns[1].Visible = false;
                    //dataGridViewChild.Columns[2].Width = 100;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void dataGridViewParent_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }

        private void dataGridViewChild_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }

        private void refreshTable(SqlConnection connection)
        {
            childAdapter.SelectCommand.Connection = connection;
            if (ds.Tables.Contains("Profiles"))
                ds.Tables["Profiles"].Clear();
            childAdapter.Fill(ds, "Profiles");
        }

        private void buttonAdd_Click(object sender, EventArgs e)
        {
            if (dataGridViewChild.CurrentRow == null || dataGridViewParent.CurrentRow == null)
            {
                return;
            }

            try
            {
                int id = int.Parse(dataGridViewParent.CurrentRow.Cells["idUser"].FormattedValue.ToString());
                string profileName = dataGridViewChild.CurrentRow.Cells["profileName"].FormattedValue.ToString();

                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    childAdapter.InsertCommand = new SqlCommand("INSERT INTO Profiles(idUser, profileName) VALUES (@idUser, @profileName)", connection);
                    childAdapter.InsertCommand.Parameters.AddWithValue("@idUser", id);
                    childAdapter.InsertCommand.Parameters.AddWithValue("@profileName", profileName);
                    childAdapter.InsertCommand.ExecuteNonQuery();
                    refreshTable(connection);
                    MessageBox.Show("Profil adaugat cu succes", "Notificare", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void buttonDelete_Click(object sender, EventArgs e)
        {
            if (dataGridViewChild.CurrentRow == null)
            {
                return;
            }

            try
            {
                string id = dataGridViewChild.CurrentRow.Cells["idProfile"].FormattedValue.ToString();
                string name = dataGridViewChild.CurrentRow.Cells["profileName"].FormattedValue.ToString();
                DialogResult dialogResult = MessageBox.Show("Stergeti profilul " + name + "?", "Confirmare stergere", MessageBoxButtons.OKCancel);
                if (dialogResult != DialogResult.OK) { return; }

                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    childAdapter.DeleteCommand = new SqlCommand("DELETE FROM Profiles WHERE idProfile = @id", connection);
                    childAdapter.DeleteCommand.Parameters.AddWithValue("@id", id);
                    childAdapter.DeleteCommand.ExecuteNonQuery();
                    refreshTable(connection);
                    MessageBox.Show("Profil stears cu succes", "Notificare", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void buttonUpdate_Click(object sender, EventArgs e)
        {
            if (dataGridViewChild.CurrentRow == null)
            {
                return;
            }

            try
            {
                int id = int.Parse(dataGridViewChild.CurrentRow.Cells["idProfile"].FormattedValue.ToString());
                string profileName = dataGridViewChild.CurrentRow.Cells["profileName"].FormattedValue.ToString();

                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    childAdapter.UpdateCommand = new SqlCommand("UPDATE Profiles SET profileName=@profileName WHERE idProfile=@id", connection);
                    childAdapter.UpdateCommand.Parameters.AddWithValue("@id", id);
                    childAdapter.UpdateCommand.Parameters.AddWithValue("@profileName", profileName);
                    childAdapter.UpdateCommand.ExecuteNonQuery();
                    refreshTable(connection);
                    MessageBox.Show("Profil actualizat cu succes", "Notificare", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
    }
}