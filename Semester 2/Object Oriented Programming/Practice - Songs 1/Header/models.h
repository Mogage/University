#pragma once

#include "domain.h"
#include <qtableview.h>
#include <vector>

class TableModel : public QAbstractTableModel
{
	Q_OBJECT
private:
	std::vector < Song > Songs;
public:
	TableModel(std::vector < Song > _Songs) : Songs{ _Songs } {}

	int rowCount(const QModelIndex& parent = QModelIndex()) const override
	{
		return Songs.size();
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
				case 1:
					return "Titlu";
				case 2:
					return "Artist";
				case 3:
					return "Rank";
				case 4:
					return "Nr melodii rank";
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
			Song aux = Songs[index.row()];
			switch (index.column())
			{
			case 0:
				return QString::number(aux.id());
			case 1:
				return QString::fromStdString(aux.title());
			case 2:
				return QString::fromStdString(aux.artist());
			case 3:
				return QString::number(aux.rank());
			case 4:
				int nrSongs = 0;
				for (auto& song : Songs)
				{
					if (song.rank() == aux.rank())
						nrSongs++;
				}
				return QString::number(nrSongs);
			}
		}
		return QVariant();
	}

	void setSongs(std::vector < Song > _Songs)
	{
		Songs = _Songs;
		auto topLeft = createIndex(0, 0);
		auto bottomRight = createIndex(rowCount(), columnCount());
		emit dataChanged(topLeft, bottomRight);
		emit layoutChanged();
	}
};