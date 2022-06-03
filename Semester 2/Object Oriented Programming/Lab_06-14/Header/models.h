#pragma once

#include "domain.h"
#include <qstring.h>
#include <qlistview.h>

class ListModel : public QAbstractListModel
{
private:
	std::vector < Product > Products;
public:
	ListModel(std::vector< Product > _Products) : Products{ _Products } {}

	int rowCount(const QModelIndex& parent = QModelIndex()) const override {
		return Products.size();
	}

	QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override {
		if (role == Qt::DisplayRole) {
			auto sp = Products[index.row()].GetName();
			int id = Products[index.row()].GetId();
			return QString::number(id) + " " + QString::fromStdString(sp);
		}
		if (role == Qt::UserRole) {
			auto tp = Products[index.row()].GetType();
			return QString::fromStdString(tp);
		}
		if (role == Qt::BackgroundRole) {
			if (Products[index.row()].GetName()[0] == 'R') {
				return QColor(Qt::red);
			}
		}
		return QVariant{};
	}

	void setProducts(const std::vector<Product>& _Products) {
		this->Products = _Products;
		auto topLeft = createIndex(0, 0);
		auto bottomR = createIndex(rowCount(), 0);
		emit dataChanged(topLeft, bottomR);
	}
};