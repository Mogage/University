using System.Configuration;
using System.Data;
using Microsoft.Data.SqlClient;

namespace PracticalExam
{
    public partial class Form1 : Form
    {
        string connectionString = @"Server=DESKTOP-P6FMEQ4;Database=S22023;Integrated Security=true;TrustServerCertificate=true;";
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
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                connection.Open();
                parentAdapter.SelectCommand = new SqlCommand("SELECT * FROM Companii", connection);
                childAdapter.SelectCommand = new SqlCommand("SELECT * FROM Concursuri", connection);
                parentAdapter.Fill(ds, "Companii");
                childAdapter.Fill(ds, "Concursuri");
                parentBS.DataSource = ds.Tables["Companii"];
                dataGridViewParent.DataSource = parentBS;
                DataColumn parentColumn = ds.Tables["Companii"].Columns["cod_companie"];
                DataColumn childColumn = ds.Tables["Concursuri"].Columns["cod_companie"];
                DataRelation relation = new DataRelation("FK_Companii_Concursuri", parentColumn, childColumn);
                ds.Relations.Add(relation);
                childBS.DataSource = parentBS;
                childBS.DataMember = "FK_Companii_Concursuri";
                dataGridViewChild.DataSource = childBS;
            }
        }

        private void refreshTable(SqlConnection connection)
        {
            childAdapter.SelectCommand.Connection = connection;
            if (ds.Tables.Contains("Concursuri"))
                ds.Tables["Concursuri"].Clear();
            childAdapter.Fill(ds, "Concursuri");
        }

        private void buttonAdd_Click(object sender, EventArgs e)
        {
            if (dataGridViewChild.CurrentRow == null || dataGridViewParent.CurrentRow == null)
            {
                return;
            }

            try
            {
                int id = int.Parse(dataGridViewParent.CurrentRow.Cells["cod_companie"].FormattedValue.ToString());
                string numeConcurs = dataGridViewChild.CurrentRow.Cells["nume_concurs"].FormattedValue.ToString();
                string descriereConcurs = dataGridViewChild.CurrentRow.Cells["descriere_concurs"].FormattedValue.ToString();
                string dataInceput = dataGridViewChild.CurrentRow.Cells["data_inceput"].FormattedValue.ToString();
                string dataSfarsit = dataGridViewChild.CurrentRow.Cells["data_sfarsit"].FormattedValue.ToString();

                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    childAdapter.InsertCommand = new SqlCommand("INSERT INTO Concursuri(nume_concurs, descriere_concurs, data_inceput, data_sfarsit, cod_companie)" +
                        " VALUES (@nume_concurs, @descriere_concurs, @data_inceput, @data_sfarsit, @cod_companie)", connection);
                    childAdapter.InsertCommand.Parameters.AddWithValue("@cod_companie", id);
                    childAdapter.InsertCommand.Parameters.AddWithValue("@nume_concurs", numeConcurs);
                    childAdapter.InsertCommand.Parameters.AddWithValue("@descriere_concurs", descriereConcurs);
                    childAdapter.InsertCommand.Parameters.AddWithValue("data_inceput", dataInceput);
                    childAdapter.InsertCommand.Parameters.AddWithValue("data_sfarsit", dataSfarsit);
                    childAdapter.InsertCommand.ExecuteNonQuery();
                    refreshTable(connection);
                    MessageBox.Show("Concurs adaugat cu succes", "Notificare", MessageBoxButtons.OK, MessageBoxIcon.Information);
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
                string id = dataGridViewChild.CurrentRow.Cells["cod_concurs"].FormattedValue.ToString();
                string name = dataGridViewChild.CurrentRow.Cells["nume_concurs"].FormattedValue.ToString();
                DialogResult dialogResult = MessageBox.Show("Stergeti concursul " + name + "?", "Confirmare stergere", MessageBoxButtons.OKCancel);
                if (dialogResult != DialogResult.OK) { return; }

                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    childAdapter.DeleteCommand = new SqlCommand("DELETE FROM Concursuri WHERE cod_concurs = @id", connection);
                    childAdapter.DeleteCommand.Parameters.AddWithValue("@id", id);
                    childAdapter.DeleteCommand.ExecuteNonQuery();
                    refreshTable(connection);
                    MessageBox.Show("Concurs sters cu succes", "Notificare", MessageBoxButtons.OK, MessageBoxIcon.Information);
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
                int id = int.Parse(dataGridViewChild.CurrentRow.Cells["cod_concurs"].FormattedValue.ToString());
                string numeConcurs = dataGridViewChild.CurrentRow.Cells["nume_concurs"].FormattedValue.ToString();
                string descriereConcurs = dataGridViewChild.CurrentRow.Cells["descriere_concurs"].FormattedValue.ToString();
                string dataInceput = dataGridViewChild.CurrentRow.Cells["data_inceput"].FormattedValue.ToString();
                string dataSfarsit = dataGridViewChild.CurrentRow.Cells["data_sfarsit"].FormattedValue.ToString();

                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    childAdapter.UpdateCommand = new SqlCommand("UPDATE Concursuri SET nume_concurs=@nume_concurs, descriere_concurs=@descriere_concurs," +
                        " data_inceput=@data_inceput, data_sfarsit=@data_sfarsit  WHERE cod_concurs=@id", connection);
                    childAdapter.UpdateCommand.Parameters.AddWithValue("@id", id);
                    childAdapter.UpdateCommand.Parameters.AddWithValue("@nume_concurs", numeConcurs);
                    childAdapter.UpdateCommand.Parameters.AddWithValue("@descriere_concurs", descriereConcurs);
                    childAdapter.UpdateCommand.Parameters.AddWithValue("data_inceput", dataInceput);
                    childAdapter.UpdateCommand.Parameters.AddWithValue("data_sfarsit", dataSfarsit);
                    childAdapter.UpdateCommand.ExecuteNonQuery();
                    refreshTable(connection);
                    MessageBox.Show("Concurs actualizat cu succes", "Notificare", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

    }
}