#pragma once

#include "domain.h"
#include <qtableview.h>

class TableModel : public QAbstractTableModel
{
	//Q_OBJECT
private:
	vector < Game >& Games;
public:
	TableModel(vector < Game >& _Games) : Games{ _Games } {}

	int rowCount(const QModelIndex& parent = QModelIndex()) const override 
	{
		return Games.size();
	}

	int columnCount(const QModelIndex& parent = QModelIndex()) const override 
	{
		return 4;
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
					return "Dimensiune";
					break;
				case 2:
					return "Jucator curent";
					break;
				case 3:
					return "Stare";
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
			Game aux = Games[index.row()];
			switch (index.column())
			{
			case 0:
				return QString::number(aux.id());
			case 1:
				return QString::number(aux.dim());
			case 2:
				return QString(aux.curent());
			case 3:
				return QString::fromStdString(aux.stare());
			}
		}
		if (Qt::UserRole == role)
		{
			int id = Games[index.row()].id();
			return QString::number(id);
		}
		return QVariant(); 
	}

	void setGames(vector < Game >& _Games)
	{
		this->Games = _Games;
		auto topLeft = createIndex(0, 0);
		auto bottomRight = createIndex(rowCount(), columnCount());
		emit dataChanged(topLeft, bottomRight);
		emit layoutChanged();
	}
};