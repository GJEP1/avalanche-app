import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib"
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import ImageLayer from 'ol/layer/Image'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import ImageWMS from 'ol/source/ImageWMS'
import ImageStatic from 'ol/source/ImageStatic'
import XYZ from 'ol/source/XYZ'
import { defaults as defaultControls, ScaleLine } from 'ol/control'
import { Draw, Modify, Snap, Select } from 'ol/interaction'
import { register } from 'ol/proj/proj4'
import proj4 from 'proj4'
import GeoJSON from 'ol/format/GeoJSON'
import { Style, Fill, Stroke, Circle as CircleStyle } from 'ol/style'
import { click } from 'ol/events/condition'
import 'ol/ol.css'

// Inject CSS for grayscale layer filter
const grayscaleStyle = document.createElement('style')
grayscaleStyle.textContent = `
  .grayscale-layer canvas {
    filter: grayscale(100%) contrast(1.1);
  }
`
document.head.appendChild(grayscaleStyle)

// Define EPSG:25833 (UTM33N for Norway)
proj4.defs('EPSG:25833', '+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
register(proj4)

// Styles for different feature types
const STYLES = {
  scenario: new Style({
    fill: new Fill({ color: 'rgba(255, 120, 0, 0.3)' }),
    stroke: new Stroke({ color: '#ff7800', width: 2 })
  }),
  scenarioSelected: new Style({
    fill: new Fill({ color: 'rgba(255, 120, 0, 0.5)' }),
    stroke: new Stroke({ color: '#ff7800', width: 3 })
  }),
  xsection: new Style({
    stroke: new Stroke({ color: '#0066ff', width: 3 })
  }),
  xsectionSelected: new Style({
    stroke: new Stroke({ color: '#0066ff', width: 4 })
  }),
  drawing: new Style({
    fill: new Fill({ color: 'rgba(0, 200, 0, 0.3)' }),
    stroke: new Stroke({ color: '#00c800', width: 2, lineDash: [5, 5] }),
    image: new CircleStyle({
      radius: 5,
      fill: new Fill({ color: '#00c800' })
    })
  }),
  vertex: new Style({
    image: new CircleStyle({
      radius: 6,
      fill: new Fill({ color: 'white' }),
      stroke: new Stroke({ color: '#0066ff', width: 2 })
    })
  })
}

// Toolbar button component
const ToolButton = ({ active, onClick, title, children }) => (
  <button
    onClick={onClick}
    title={title}
    style={{
      width: 36,
      height: 36,
      border: 'none',
      background: active ? '#e3f2fd' : 'white',
      color: active ? '#1976d2' : '#333',
      cursor: 'pointer',
      borderRadius: 4,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      transition: 'background 0.2s'
    }}
  >
    {children}
  </button>
)

// SVG Icons
const Icons = {
  navigate: (
    <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
      <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/>
    </svg>
  ),
  select: (
    <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
      <path d="M3 5h2V3H3v2zm0 8h2v-2H3v2zm4 8h2v-2H7v2zM3 9h2V7H3v2zm10-6h-2v2h2V3zm6 0v2h2V3h-2zM5 21v-2H3v2h2zm-2-4h2v-2H3v2zM9 3H7v2h2V3zm2 18h2v-2h-2v2zm8-8h2v-2h-2v2zm0 8v-2h-2v2h2zm0-12h2V7h-2v2zm0 8h2v-2h-2v2zm-4 4h2v-2h-2v2zm0-16h2V3h-2v2z"/>
    </svg>
  ),
  polygon: (
    <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
      <path d="M2 2v8h2V4h6V2H2zm20 0h-8v2h6v6h2V2zM2 14v8h8v-2H4v-6H2zm20 8h-8v-2h6v-6h2v8z"/>
      <path d="M6 6h12v12H6z" fill="none" stroke="currentColor" strokeWidth="2"/>
    </svg>
  ),
  line: (
    <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
      <path d="M3.5 18.5l6-6 4 4 8-8" fill="none" stroke="currentColor" strokeWidth="2"/>
      <circle cx="3.5" cy="18.5" r="2"/>
      <circle cx="9.5" cy="12.5" r="2"/>
      <circle cx="13.5" cy="16.5" r="2"/>
      <circle cx="21.5" cy="8.5" r="2"/>
    </svg>
  ),
  modify: (
    <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
      <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
    </svg>
  ),
  delete: (
    <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
      <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
    </svg>
  ),
  clear: (
    <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
      <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
    </svg>
  ),
  fit: (
    <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
      <path d="M15 3l2.3 2.3-2.89 2.87 1.42 1.42L18.7 6.7 21 9V3h-6zM3 9l2.3-2.3 2.87 2.89 1.42-1.42L6.7 5.3 9 3H3v6zm6 12l-2.3-2.3 2.89-2.87-1.42-1.42L5.3 17.3 3 15v6h6zm12-6l-2.3 2.3-2.87-2.89-1.42 1.42 2.89 2.87L15 21h6v-6z"/>
    </svg>
  )
}

const OlMapComponent = (props) => {
  const mapRef = useRef(null)
  const mapInstanceRef = useRef(null)
  const drawInteractionRef = useRef(null)
  const modifyInteractionRef = useRef(null)
  const selectInteractionRef = useRef(null)
  const snapInteractionRef = useRef(null)

  // Vector sources
  const scenarioSourceRef = useRef(null)
  const xsectionSourceRef = useRef(null)
  const drawingSourceRef = useRef(null)
  const rasterLayerRef = useRef(null)

  const [currentTool, setCurrentTool] = useState('navigate')
  const [selectedFeature, setSelectedFeature] = useState(null)
  const [selectedFeatureLayer, setSelectedFeatureLayer] = useState(null)  // 'drawings', 'scenarios', or 'xsections'

  const {
    center = [61.5, 8.5],
    zoom = 7,
    basemap = 'Grunnkart',
    showHillshade = true,
    enableDrawing = false,
    showToolbar = true,
    uploadedFeatures = null,
    scenarioFeatures = null,
    xsectionFeatures = null,
    rasterOverlay = null,
    zoomToBounds = null,
    height = 600
  } = props.args || {}

  // Get all drawn features as GeoJSON
  const getAllDrawings = useCallback(() => {
    if (!drawingSourceRef.current) return null
    const features = drawingSourceRef.current.getFeatures()
    if (features.length === 0) return null

    const geojson = new GeoJSON().writeFeaturesObject(features, {
      dataProjection: 'EPSG:25833',
      featureProjection: 'EPSG:25833'
    })
    return geojson
  }, [])

  // Send state to Streamlit
  const sendToStreamlit = useCallback((eventType, data = null, layerType = null) => {
    Streamlit.setComponentValue({
      type: eventType,
      feature: data,
      allDrawings: getAllDrawings(),
      selectedFeatureId: selectedFeature?.ol_uid || null,
      selectedFeatureLayer: layerType || selectedFeatureLayer
    })
  }, [getAllDrawings, selectedFeature, selectedFeatureLayer])

  // Clear all interactions
  const clearInteractions = useCallback(() => {
    const map = mapInstanceRef.current
    if (!map) return

    if (drawInteractionRef.current) {
      map.removeInteraction(drawInteractionRef.current)
      drawInteractionRef.current = null
    }
    if (modifyInteractionRef.current) {
      map.removeInteraction(modifyInteractionRef.current)
      modifyInteractionRef.current = null
    }
    if (selectInteractionRef.current) {
      map.removeInteraction(selectInteractionRef.current)
      selectInteractionRef.current = null
    }
    if (snapInteractionRef.current) {
      map.removeInteraction(snapInteractionRef.current)
      snapInteractionRef.current = null
    }
  }, [])

  // Set active tool
  const setTool = useCallback((tool) => {
    setCurrentTool(tool)
    clearInteractions()

    const map = mapInstanceRef.current
    if (!map || !drawingSourceRef.current) return

    switch (tool) {
      case 'select':
        const select = new Select({
          // Allow selecting from all feature layers (drawings, scenarios, xsections)
          layers: (layer) => {
            const name = layer.get('name')
            return name === 'drawings' || name === 'scenarios' || name === 'xsections'
          },
          condition: click,
          style: (feature) => {
            // Use highlighted style for selected features
            const featureType = feature.get('featureType')
            if (featureType === 'xsection') return STYLES.xsectionSelected
            return STYLES.scenarioSelected
          }
        })
        select.on('select', (e) => {
          if (e.selected.length > 0) {
            const feature = e.selected[0]
            setSelectedFeature(feature)

            // Determine which layer the feature belongs to
            const featureType = feature.get('featureType')
            let layerType = 'drawings'

            // Check if feature is from scenarios or xsections source
            if (scenarioSourceRef.current && scenarioSourceRef.current.hasFeature(feature)) {
              layerType = 'scenarios'
            } else if (xsectionSourceRef.current && xsectionSourceRef.current.hasFeature(feature)) {
              layerType = 'xsections'
            }

            setSelectedFeatureLayer(layerType)
            sendToStreamlit('feature_selected', new GeoJSON().writeFeatureObject(feature, {
              dataProjection: 'EPSG:25833',
              featureProjection: 'EPSG:25833'
            }), layerType)
          } else {
            setSelectedFeature(null)
            setSelectedFeatureLayer(null)
            sendToStreamlit('feature_deselected')
          }
        })
        map.addInteraction(select)
        selectInteractionRef.current = select
        break

      case 'polygon':
        const drawPolygon = new Draw({
          source: drawingSourceRef.current,
          type: 'Polygon',
          style: STYLES.drawing
        })
        drawPolygon.on('drawend', (e) => {
          const feature = e.feature
          feature.set('featureType', 'scenario')
          feature.set('name', `scenario_${Date.now()}`)
          // Use setTimeout to ensure feature is added to source before getting all drawings
          setTimeout(() => {
            sendToStreamlit('feature_drawn', new GeoJSON().writeFeatureObject(feature, {
              dataProjection: 'EPSG:25833',
              featureProjection: 'EPSG:25833'
            }))
          }, 10)
        })
        map.addInteraction(drawPolygon)
        drawInteractionRef.current = drawPolygon

        const snapPoly = new Snap({ source: drawingSourceRef.current })
        map.addInteraction(snapPoly)
        snapInteractionRef.current = snapPoly
        break

      case 'line':
        const drawLine = new Draw({
          source: drawingSourceRef.current,
          type: 'LineString',
          style: STYLES.drawing
        })
        drawLine.on('drawend', (e) => {
          const feature = e.feature
          feature.set('featureType', 'xsection')
          feature.set('name', `xsection_${Date.now()}`)
          // Use setTimeout to ensure feature is added to source before getting all drawings
          setTimeout(() => {
            sendToStreamlit('feature_drawn', new GeoJSON().writeFeatureObject(feature, {
              dataProjection: 'EPSG:25833',
              featureProjection: 'EPSG:25833'
            }))
          }, 10)
        })
        map.addInteraction(drawLine)
        drawInteractionRef.current = drawLine

        const snapLine = new Snap({ source: drawingSourceRef.current })
        map.addInteraction(snapLine)
        snapInteractionRef.current = snapLine
        break

      case 'modify':
        const modify = new Modify({
          source: drawingSourceRef.current,
          style: STYLES.vertex
        })
        modify.on('modifyend', () => {
          sendToStreamlit('features_modified')
        })
        map.addInteraction(modify)
        modifyInteractionRef.current = modify

        const snapMod = new Snap({ source: drawingSourceRef.current })
        map.addInteraction(snapMod)
        snapInteractionRef.current = snapMod
        break

      case 'navigate':
      default:
        break
    }
  }, [clearInteractions, sendToStreamlit])

  // Delete selected feature (only from drawings layer, not project features)
  const deleteSelected = useCallback(() => {
    if (selectedFeature && drawingSourceRef.current && selectedFeatureLayer === 'drawings') {
      drawingSourceRef.current.removeFeature(selectedFeature)
      setSelectedFeature(null)
      setSelectedFeatureLayer(null)
      sendToStreamlit('feature_deleted')
    } else if (selectedFeature && (selectedFeatureLayer === 'scenarios' || selectedFeatureLayer === 'xsections')) {
      // Inform user that project features must be deleted from sidebar
      sendToStreamlit('delete_project_feature_requested', new GeoJSON().writeFeatureObject(selectedFeature, {
        dataProjection: 'EPSG:25833',
        featureProjection: 'EPSG:25833'
      }), selectedFeatureLayer)
    }
  }, [selectedFeature, selectedFeatureLayer, sendToStreamlit])

  // Clear all drawings
  const clearAll = useCallback(() => {
    if (drawingSourceRef.current) {
      drawingSourceRef.current.clear()
      setSelectedFeature(null)
      sendToStreamlit('drawings_cleared')
    }
  }, [sendToStreamlit])

  // Fit to all features
  const fitToFeatures = useCallback(() => {
    const map = mapInstanceRef.current
    if (!map) return

    const sources = [scenarioSourceRef.current, xsectionSourceRef.current, drawingSourceRef.current]
    let hasFeatures = false

    // Create combined extent
    let extent = null
    sources.forEach(source => {
      if (source && source.getFeatures().length > 0) {
        hasFeatures = true
        const sourceExtent = source.getExtent()
        if (!extent) {
          extent = [...sourceExtent]
        } else {
          extent[0] = Math.min(extent[0], sourceExtent[0])
          extent[1] = Math.min(extent[1], sourceExtent[1])
          extent[2] = Math.max(extent[2], sourceExtent[2])
          extent[3] = Math.max(extent[3], sourceExtent[3])
        }
      }
    })

    if (hasFeatures && extent) {
      map.getView().fit(extent, { padding: [50, 50, 50, 50], maxZoom: 16 })
    }
  }, [])

  // Initialize map
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return

    const centerUTM = proj4('EPSG:4326', 'EPSG:25833', [center[1], center[0]])

    // Create vector sources
    const scenarioSource = new VectorSource()
    const xsectionSource = new VectorSource()
    const drawingSource = new VectorSource()

    scenarioSourceRef.current = scenarioSource
    xsectionSourceRef.current = xsectionSource
    drawingSourceRef.current = drawingSource

    // Create vector layers with styling
    const scenarioLayer = new VectorLayer({
      source: scenarioSource,
      name: 'scenarios',
      style: STYLES.scenario
    })

    const xsectionLayer = new VectorLayer({
      source: xsectionSource,
      name: 'xsections',
      style: STYLES.xsection
    })

    const drawingLayer = new VectorLayer({
      source: drawingSource,
      name: 'drawings',
      style: (feature) => {
        const type = feature.get('featureType')
        if (type === 'xsection') return STYLES.xsection
        return STYLES.scenario
      }
    })

    // Create map
    const map = new Map({
      target: mapRef.current,
      layers: [scenarioLayer, xsectionLayer, drawingLayer],
      view: new View({
        projection: 'EPSG:25833',
        center: centerUTM,
        zoom: zoom,
        maxZoom: 18,
        minZoom: 3
      }),
      controls: defaultControls().extend([new ScaleLine({ units: 'metric' })])
    })

    mapInstanceRef.current = map
    Streamlit.setFrameHeight(height)

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.setTarget(null)
      }
    }
  }, [])

  // Update basemap
  useEffect(() => {
    if (!mapInstanceRef.current) return

    const map = mapInstanceRef.current
    const layers = map.getLayers().getArray()

    // Remove base layers (keep vector layers which are last 3)
    const vectorLayers = layers.slice(-3)
    map.getLayers().clear()

    // Base layers - working services
    const baseLayers = {
      'Terrain': new ImageLayer({
        source: new ImageWMS({
          url: 'https://wms.geonorge.no/skwms1/wms.terrengmodell',
          params: { 'LAYERS': 'relieff', 'FORMAT': 'image/png' },
          projection: 'EPSG:25833',
          attributions: '© Kartverket'
        }),
        className: 'grayscale-layer'
      }),
      'Grunnkart': new ImageLayer({
        source: new ImageWMS({
          url: 'https://wms.geonorge.no/skwms1/wms.norges_grunnkart',
          params: { 'LAYERS': 'Norges_grunnkart', 'FORMAT': 'image/png' },
          projection: 'EPSG:25833',
          attributions: '© Kartverket'
        })
      }),
      'Satellite (ESRI)': new TileLayer({
        source: new XYZ({
          url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          attributions: '© Esri, Maxar, Earthstar Geographics',
          maxZoom: 19
        })
      })
    }

    map.addLayer(baseLayers[basemap] || baseLayers['Grunnkart'])

    // Add hillshade overlay (DTM relief with transparency) for terrain visualization
    // Works best with Grunnkart and Satellite
    if (showHillshade && basemap !== 'Terrain') {
      const hillshadeLayer = new ImageLayer({
        source: new ImageWMS({
          url: 'https://wms.geonorge.no/skwms1/wms.terrengmodell',
          params: { 'LAYERS': 'relieff', 'FORMAT': 'image/png' },
          projection: 'EPSG:25833'
        }),
        opacity: 0.35  // Semi-transparent overlay
      })
      map.addLayer(hillshadeLayer)
    }

    // Re-add vector layers
    vectorLayers.forEach(layer => map.addLayer(layer))
  }, [basemap, showHillshade])

  // Update raster overlay
  useEffect(() => {
    if (!mapInstanceRef.current) return

    const map = mapInstanceRef.current

    if (rasterLayerRef.current) {
      map.removeLayer(rasterLayerRef.current)
      rasterLayerRef.current = null
    }

    if (rasterOverlay && rasterOverlay.image && rasterOverlay.bounds) {
      const [minx, miny, maxx, maxy] = rasterOverlay.bounds

      const rasterLayer = new ImageLayer({
        source: new ImageStatic({
          url: rasterOverlay.image,
          imageExtent: [minx, miny, maxx, maxy],
          projection: 'EPSG:25833'
        }),
        opacity: 0.7
      })

      const layers = map.getLayers()
      layers.insertAt(layers.getLength() - 3, rasterLayer)
      rasterLayerRef.current = rasterLayer
    }
  }, [rasterOverlay])

  // Update scenario features
  useEffect(() => {
    if (!scenarioSourceRef.current) return

    scenarioSourceRef.current.clear()

    if (scenarioFeatures && scenarioFeatures.features) {
      const features = new GeoJSON().readFeatures(scenarioFeatures, {
        dataProjection: 'EPSG:25833',
        featureProjection: 'EPSG:25833'
      })
      features.forEach(f => f.set('featureType', 'scenario'))
      scenarioSourceRef.current.addFeatures(features)
    }
  }, [scenarioFeatures])

  // Update xsection features
  useEffect(() => {
    if (!xsectionSourceRef.current) return

    xsectionSourceRef.current.clear()

    if (xsectionFeatures && xsectionFeatures.features) {
      const features = new GeoJSON().readFeatures(xsectionFeatures, {
        dataProjection: 'EPSG:25833',
        featureProjection: 'EPSG:25833'
      })
      features.forEach(f => f.set('featureType', 'xsection'))
      xsectionSourceRef.current.addFeatures(features)
    }
  }, [xsectionFeatures])

  // Handle uploaded features (legacy support)
  useEffect(() => {
    if (!drawingSourceRef.current || !uploadedFeatures) return

    drawingSourceRef.current.clear()

    if (uploadedFeatures.features) {
      const features = new GeoJSON().readFeatures(uploadedFeatures, {
        dataProjection: 'EPSG:25833',
        featureProjection: 'EPSG:25833'
      })
      drawingSourceRef.current.addFeatures(features)
    }
  }, [uploadedFeatures])

  // Handle zoom to bounds
  useEffect(() => {
    if (!mapInstanceRef.current || !zoomToBounds) return

    const bounds = zoomToBounds.bounds || zoomToBounds
    if (Array.isArray(bounds) && bounds.length === 4) {
      mapInstanceRef.current.getView().fit(bounds, {
        padding: [50, 50, 50, 50],
        duration: 500
      })
    }
  }, [zoomToBounds])

  // Initialize with navigate tool
  useEffect(() => {
    if (mapInstanceRef.current && enableDrawing) {
      setTool('navigate')
    }
  }, [enableDrawing, setTool])

  // Toolbar component
  const Toolbar = () => (
    <div style={{
      position: 'absolute',
      top: 10,
      left: 50,
      zIndex: 1000,
      background: 'white',
      borderRadius: 4,
      boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
      display: 'flex',
      gap: 2,
      padding: 4
    }}>
      <ToolButton active={currentTool === 'navigate'} onClick={() => setTool('navigate')} title="Navigate">
        {Icons.navigate}
      </ToolButton>
      <ToolButton active={currentTool === 'select'} onClick={() => setTool('select')} title="Select features">
        {Icons.select}
      </ToolButton>
      <div style={{ width: 1, background: '#ddd', margin: '4px' }} />
      <ToolButton active={currentTool === 'polygon'} onClick={() => setTool('polygon')} title="Draw scenario polygon">
        {Icons.polygon}
      </ToolButton>
      <ToolButton active={currentTool === 'line'} onClick={() => setTool('line')} title="Draw cross-section line">
        {Icons.line}
      </ToolButton>
      <ToolButton active={currentTool === 'modify'} onClick={() => setTool('modify')} title="Edit features">
        {Icons.modify}
      </ToolButton>
      <div style={{ width: 1, background: '#ddd', margin: '4px' }} />
      <ToolButton onClick={deleteSelected} title="Delete selected">
        {Icons.delete}
      </ToolButton>
      <ToolButton onClick={clearAll} title="Clear all drawings">
        {Icons.clear}
      </ToolButton>
      <div style={{ width: 1, background: '#ddd', margin: '4px' }} />
      <ToolButton onClick={fitToFeatures} title="Fit to features">
        {Icons.fit}
      </ToolButton>
    </div>
  )

  return (
    <div style={{ position: 'relative', width: '100%', height: `${height}px` }}>
      <div
        ref={mapRef}
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: '#f0f0f0'
        }}
      />
      {showToolbar && enableDrawing && <Toolbar />}
    </div>
  )
}

export default withStreamlitConnection(OlMapComponent)
