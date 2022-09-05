import pandas as pd
import numpy as np
import folium
from folium import plugins

def load_grid_locations(path):
    grid_locations=pd.read_csv(path)
    return grid_locations


class MakeMap:

    def __init__(self, grid_locations, data, lat_colname, lon_colname, y_pred_lat_colname, y_pred_lon_colname, date_colname):
        self.data=data
        self.grid_locations=grid_locations
        self.lat = lat_colname
        self.lon = lon_colname
        self.y_pred_lat = y_pred_lat_colname
        self.y_pred_lon = y_pred_lon_colname
        self.date = date_colname        

    def initialize_map(self):
        self.m = folium.Map([35.10734569104119, 129.09589426708501], zoom_start=16)
    
    def get_styles(self):
        self.berth_style = {"color":"#23A7D8","fill_color":"#44CCFF"}
        self.block_style = {"color":'#ff7800', "fill_color":'#ffff00'}
        self.road_style  = {"color":"#7C7C7C", "fill_color":"#9C9C9C"}

    def draw_grid(self):
        for idx, row in self.grid_locations.dropna(subset=["S_latitude","S_longitude","E_latitude","E_longitude"]).iterrows():
            s,e=row[["S_latitude","S_longitude"]].values,row[["E_latitude","E_longitude"]].values
        
            if row.Group == "Berth":
                style=self.berth_style
                
            else: style=self.block_style
            
            if row.Group == "Road":
                style=self.road_style
                
            folium.Rectangle(
                bounds=[s,e], 
                color=style["color"], 
                fill=True, 
                fill_color=style["fill_color"], 
                fill_opacity=0.5, 
                line_width=0.1, 
                weight=0.5,
                tooltip=row.Mapping,
                bubbling_mouse_events=True).add_to(self.m)


    def draw_point(self,opacity, isin_predict=False, add_last_point=[True,False]):
        

        actual_lines=[
        {
        "coordinates":[row[self.lon], row[self.lat]],
        "dates":[row[self.date]],
        "color":"red" if idx%6 != 0 else "black"
        } 
        for idx, (_, row) in enumerate(self.data.reset_index(drop=True).iterrows(), start=1)
        ]

        actual_features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": line["coordinates"],
                    "popup":str(line["dates"])
                },
                "properties": {
                    "times": line["dates"],
                    "icon":"circle",
                    "style": {
                        "radius" : 0.5,
                        "color" : line["color"],
                        "weight" : line["weight"] if "weight" in line else 5,
                        "opacity": opacity
                        # "tooltip" : line["dates"],
                        # "popup" : line["dates"]
                    },
                },
            }
            for line in actual_lines
        ]

        plugins.TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": actual_features,
            },
            period="PT1S",
            add_last_point=add_last_point,
        ).add_to(self.m)
        
        if isin_predict==True:

            pred_lines=[
                    {
                    "coordinates":[row[self.y_pred_lon], row[self.y_pred_lat]],
                    "dates":[row[self.date]],
                    "color":"green"
                    } 
                    for idx, (_, row) in enumerate(self.data[self.data["longitude_y_pred"].isnull()==False].reset_index(drop=True).iterrows())
            ]

            pred_features=[
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": line["coordinates"],
                        "popup":str(line["dates"])
                    },
                    "properties": {
                        "times": line["dates"],
                        "icon":"circle",
                        "style": {
                            "radius" : 0.5,
                            "color" : line["color"],
                            "weight" : line["weight"] if "weight" in line else 5,
                            "opacity": 0.8
                            # "tooltip" : line["dates"],
                            # "popup" : line["dates"]
                        },
                    },
                } for line in pred_lines
            ]

            plugins.TimestampedGeoJson(
                {
                    "type": "FeatureCollection",
                    "features": pred_features,
                },
                period="PT1S",
                add_last_point=add_last_point
            ).add_to(self.m)
                   
    def save(self, path):
        self.m.save(path)

    def return_map(self):
        return self.m