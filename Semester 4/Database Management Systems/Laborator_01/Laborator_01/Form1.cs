using Microsoft.Data.SqlClient;
using System.Data;

namespace Laborator_01
{
    public partial class Form1 : Form
    {

        string connectionString = @"Server=DESKTOP-P6FMEQ4;Database=SGBD_01;Integrated Security=true;TrustServerCertificate=true";
        DataSet dataSet = new DataSet();
        SqlDataAdapter adapter= new SqlDataAdapter();
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {
                using(SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();
                    adapter.SelectCommand = new SqlCommand("SELECT * FROM Joburi", connection);
                    adapter.Fill(dataSet, "Joburi");
                    dataGridView1.DataSource = dataSet.Tables["Joburi"];
                }
            }
            catch(Exception ex) 
            {
                MessageBox.Show(ex.Message);
            }  
        }

        private void button1_Click(object sender, EventArgs e)
        {
            try
            {
                using(SqlConnection connection = new SqlConnection(connectionString))
                {
                    adapter.SelectCommand.Connection = connection;
                    if (dataSet.Tables.Contains("Joburi"))
                        dataSet.Tables["Joburi"].Clear();
                    adapter.Fill(dataSet, "Joburi");

                }
            }
            catch(Exception ex) 
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

    }
}