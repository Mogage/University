#pragma once

#include "domain.h"
#include <qtableview.h>
#include <qslider.h>
#include <vector>

class TableModel : public QAbstractTableModel
{
	Q_OBJECT
private:
	std::vector < Produs > Products;
	QSlider& Slider;
public:
	TableModel(std::vector < Produs > _Products, QSlider& _Slider) : Products{ _Products }, Slider{ _Slider }{}

	int rowCount(const QModelIndex& parent = QModelIndex()) const override
	{
		return Products.size();
	}

	int columnCount(const QModelIndex& parent = QModelIndex()) const override
	{
		return 5;
	}

	QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override
	{
		if (Qt::Horizontal == orientation)
		{
			if (Qt::DisplayRole == role)
			{
				switch (section)
				{
				case 0:
					return "Id";
					break;
				case 1:
					return "Nume";
					break;
				case 2:
					return "Tip";
					break;
				case 3:
					return "Pret";
					break;
				case 4:
					return "Numar Vocale";
					break;
				default:
					break;
				}
			}
		}
		return QVariant();
	}

	QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override 
	{
		if (Qt::DisplayRole == role)
		{
			Produs aux = Products[index.row()];
			switch (index.column())
			{
			case 0:
				return QString::number(aux.id());
			case 1:
				return QString::fromStdString(aux.name());
			case 2:
				return QString::fromStdString(aux.type());
			case 3:
				return QString::number(aux.price());
			case 4:
				int nrVocale = 0;
				string name = aux.name();
				for (auto& litera : name)
				{
					if (strchr("aeiouAEIOU", litera))
						nrVocale++;
				}
				return QString::number(nrVocale);
			}
		}
		if (Qt::BackgroundRole == role)
		{
			if (Slider.value() == 50)
			{
				Produs aux = Products[index.row()];
				if (aux.price() <= 50)
					return QBrush{ Qt::red };
				return QBrush{ Qt::white };
			}
			return QBrush{ Qt::white };
		}
		if (Qt::UserRole == role)
		{
			int id = Products[index.row()].id();
			return QString::number(id);
		}
		return QVariant();
	}

	void update()
	{
		emit layoutChanged();
	}

	void setProducts(std::vector<Produs> _Products)
	{
		this->Products = _Products;
		auto topLeft = createIndex(0, 0);
		auto bottomR = createIndex(rowCount(), columnCount());
		emit dataChanged(topLeft, bottomR);
		emit layoutChanged();
	}
};