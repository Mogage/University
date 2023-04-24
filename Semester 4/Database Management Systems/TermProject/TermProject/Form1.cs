using System.ComponentModel.Design;
using System.Data;
using System.Configuration;
using Microsoft.Data.SqlClient;
using System;
using System.Reflection;

namespace TermProject
{
    public partial class Form1 : Form
    {
        DataSet ds = new DataSet();
        SqlDataAdapter parentAdapter = new SqlDataAdapter();
        SqlDataAdapter childAdapter = new SqlDataAdapter();
        BindingSource parentBS = new BindingSource();
        BindingSource childBS = new BindingSource();

        private readonly string connectionString = ConfigurationManager.AppSettings["connectionString"];
        private readonly string parent = ConfigurationManager.AppSettings["parent"];
        private readonly string parentSelect = ConfigurationManager.AppSettings["parentSelect"];
        private readonly List<string> parentParams = new List<string>(ConfigurationManager.AppSettings["parentParams"].Split(','));
        private readonly string child = ConfigurationManager.AppSettings["child"];
        private readonly string childSelect = ConfigurationManager.AppSettings["childSelect"];
        private readonly List<string> childParams = new List<string>(ConfigurationManager.AppSettings["childParams"].Split(','));
        private readonly List<string> childParamsTypes = new List<string>(ConfigurationManager.AppSettings["childParamsTypes"].Split(','));
        private readonly string insertCommand = ConfigurationManager.AppSettings["insertCommand"];
        private readonly string deleteCommand = ConfigurationManager.AppSettings["deleteCommand"];
        private readonly string updateCommand = ConfigurationManager.AppSettings["updateCommand"];

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
                    parentAdapter.SelectCommand = new SqlCommand(parentSelect, connection);
                    childAdapter.SelectCommand = new SqlCommand(childSelect, connection);
                    //parentAdapter.SelectCommand = new SqlCommand("SELECT * FROM Users", connection);
                    //childAdapter.SelectCommand = new SqlCommand("SELECT * FROM Profiles", connection);
                    parentAdapter.Fill(ds, parent);
                    childAdapter.Fill(ds, child);
                    parentBS.DataSource = ds.Tables[parent];
                    dataGridViewParent.DataSource = parentBS;
                    DataColumn parentColumn = ds.Tables[parent].Columns[parentParams[0]];
                    DataColumn childColumn = ds.Tables[child].Columns[parentParams[0]];
                    string relationName = "FK_" + parent + "_" + child;
                    DataRelation relation = new DataRelation(relationName, parentColumn, childColumn);
                    ds.Relations.Add(relation);
                    childBS.DataSource = parentBS;
                    childBS.DataMember = relationName;
                    dataGridViewChild.DataSource = childBS;
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
            if (ds.Tables.Contains(child))
                ds.Tables[child].Clear();
            childAdapter.Fill(ds, child);
        }

        private void buttonAdd_Click(object sender, EventArgs e)
        {
            if (dataGridViewChild.CurrentRow == null || dataGridViewParent.CurrentRow == null)
            {
                return;
            }

            try
            {
                int id = int.Parse(dataGridViewParent.CurrentRow.Cells[parentParams[0]].FormattedValue.ToString());
                List<string> parameters = new List<string>();
                for (int index = 2; index < childParams.Count; index++)
                {
                    parameters.Add(dataGridViewChild.CurrentRow.Cells[childParams[index]].FormattedValue.ToString());
                }

                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    childAdapter.InsertCommand = new SqlCommand(insertCommand, connection);
                    childAdapter.InsertCommand.Parameters.AddWithValue("@" + parentParams[0], id);
                    for (int index = 0; index < parameters.Count; index++)
                    {
                        childAdapter.InsertCommand.Parameters.AddWithValue("@" + childParams[index + 2], parameters[index]);
                    }
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
                string id = dataGridViewChild.CurrentRow.Cells[childParams[0]].FormattedValue.ToString();
                DialogResult dialogResult = MessageBox.Show("Stergeti profilul cu id-ul " + id + "?", "Confirmare stergere", MessageBoxButtons.OKCancel);
                if (dialogResult != DialogResult.OK) { return; }

                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    childAdapter.DeleteCommand = new SqlCommand(deleteCommand, connection);
                    childAdapter.DeleteCommand.Parameters.AddWithValue(childParams[0], id);
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
                List<string> parameters = new List<string>();
                parameters.Add(dataGridViewChild.CurrentRow.Cells[childParams[0]].FormattedValue.ToString());
                for (int index = 2; index < childParams.Count; index++)
                {
                    parameters.Add(dataGridViewChild.CurrentRow.Cells[childParams[index]].FormattedValue.ToString());
                }

                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    childAdapter.UpdateCommand = new SqlCommand(updateCommand, connection);
                    childAdapter.UpdateCommand.Parameters.AddWithValue("@" + childParams[0], int.Parse(parameters[0]));
                    for (int index = 1; index < parameters.Count; index++)
                    {
                        if (childParamsTypes[index + 1] == "int")
                        {
                            childAdapter.UpdateCommand.Parameters.AddWithValue("@" + childParams[index + 1], int.Parse(parameters[index]));
                            continue;
                        }
                        childAdapter.UpdateCommand.Parameters.AddWithValue("@" + childParams[index + 1], parameters[index]);
                    }
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