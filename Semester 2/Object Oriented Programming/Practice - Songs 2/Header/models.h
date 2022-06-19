#pragma once

#include "domain.h"
#include <vector>
#include <qtableview.h>
#include <algorithm>

using std::vector;

class TableModel : public QAbstractTableModel
{
private:
	vector < Song > Songs;
public:
	TableModel(const vector < Song >& _Songs) : Songs{ _Songs } {}

	int rowCount(const QModelIndex& parent = QModelIndex()) const override
	{
		return (int)Songs.size();
	}

	int columnCount(const QModelIndex& parent = QModelIndex()) const override
	{
		return 6;
	}

	QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override
	{
		if (orientation == Qt::Horizontal)
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
					return "Gen";
				case 4:
					return "Nr acelasi autor";
				case 5:
					return "Nr acelasi gen";
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
			int nrArtist = 0, nrGen = 0;
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
				return QString::fromStdString(aux.gen());
			case 4:
				for (const auto& song : Songs)
				{
					if (song.artist() == aux.artist())
						nrArtist++;
				}
				return QString::number(nrArtist);
			case 5:
				for (const auto& song : Songs)
				{
					if (song.gen() == aux.gen())
						nrGen++;
				}
				return QString::number(nrGen);
			default:
				break;
			}
		}
		return QVariant();
	}

	void setSongs(const vector < Song >& _Songs)
	{
		Songs = _Songs;
		std::sort(Songs.begin(), Songs.end(), [](const Song& S1, const Song& S2) { return S1.artist() < S2.artist(); });
		auto topLeft = createIndex(0, 0);
		auto bottomRight = createIndex(rowCount(), columnCount());
		emit dataChanged(topLeft, bottomRight);
		emit layoutChanged();
	}
};