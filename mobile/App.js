import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, SafeAreaView, Dimensions, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import MapView, { Marker, Polyline } from 'react-native-maps';

const { width, height } = Dimensions.get('window');

const NODES = [
  { id: 'nashik', title: 'Nashik Tomato Farm (Origin)', lat: 19.9975, lon: 73.7898, type: 'farm' },
  { id: 'vashi', title: 'Vashi APMC Mandi (Destination)', lat: 19.0771, lon: 72.9988, type: 'warehouse' },
  { id: 'distress', title: 'Pallet-T8 Event', lat: 19.5, lon: 73.5, type: 'distress' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('Map');
  const [permission, requestPermission] = useCameraPermissions();
  const [scanned, setScanned] = useState(false);
  
  const initialRegion = {
    latitude: 19.5,
    longitude: 73.4,
    latitudeDelta: 1.5,
    longitudeDelta: 1.5,
  };

  if (!permission && activeTab === 'Scan') {
    return <View style={styles.container} />;
  }
  
  if (activeTab === 'Scan' && !permission?.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
        <View style={styles.tabBar}>
          <TouchableOpacity style={styles.tabItem} onPress={() => setActiveTab('Map')}>
            <Text style={styles.tabText}>🗺️ Bio-Maps</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const handleBarCodeScanned = ({ type, data }) => {
    setScanned(true);
    Alert.alert(
      "Kinexica PINN",
      `Scanned Data:\n${data}\n\nType: ${type}\nVerified by Visual-PINN.`,
      [{ text: "OK", onPress: () => setScanned(false) }]
    );
  };

  const renderScan = () => (
    <View style={styles.fullScreen}>
      <CameraView 
        style={styles.fullScreen} 
        facing="back"
        onBarcodeScanned={scanned ? undefined : handleBarCodeScanned}
        barcodeScannerSettings={{
          barcodeTypes: ["qr", "ean13", "code128"],
        }}
      >
        <View style={styles.overlay}>
          <Text style={styles.overlayText}>Scan Kinexica QR Code</Text>
          <View style={styles.scanBox} />
        </View>
      </CameraView>
    </View>
  );

  const renderMap = () => (
    <View style={styles.fullScreen}>
      <MapView style={styles.fullScreen} initialRegion={initialRegion}>
        {NODES.map((node) => (
          <Marker 
            key={node.id}
            coordinate={{ latitude: node.lat, longitude: node.lon }}
            title={node.title}
            pinColor={node.type === 'distress' ? 'red' : (node.type === 'farm' ? 'green' : 'blue')}
          />
        ))}
        <Polyline
          coordinates={[
            { latitude: 19.9975, longitude: 73.7898 },
            { latitude: 19.5, longitude: 73.5 },
            { latitude: 19.0771, longitude: 72.9988 },
          ]}
          strokeColor="#00cc66"
          strokeWidth={4}
          lineDashPattern={[5, 5]}
        />
      </MapView>
      <View style={styles.floatingPanel}>
        <Text style={styles.panelTitle}>Supply Chain Monitoring</Text>
        <Text style={styles.panelText}>Active Dispatches: 1</Text>
        <Text style={styles.panelText}>Distress Events: 1 (Red Pin)</Text>
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      {activeTab === 'Map' ? renderMap() : renderScan()}
      
      <View style={styles.tabBar}>
        <TouchableOpacity 
          style={[styles.tabItem, activeTab === 'Map' && styles.activeTab]}
          onPress={() => setActiveTab('Map')}
        >
          <Text style={[styles.tabText, activeTab === 'Map' && styles.activeTabText]}>🗺️ Bio-Maps</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.tabItem, activeTab === 'Scan' && styles.activeTab]}
          onPress={() => setActiveTab('Scan')}
        >
          <Text style={[styles.tabText, activeTab === 'Scan' && styles.activeTabText]}>📸 Scan Hub</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#121212',
  },
  fullScreen: {
    flex: 1,
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
    marginTop: 100,
    color: '#fff',
    fontSize: 16,
  },
  button: {
    backgroundColor: '#00cc66',
    padding: 15,
    marginHorizontal: 40,
    borderRadius: 8,
    marginTop: 20,
  },
  buttonText: {
    color: '#fff',
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 16,
  },
  tabBar: {
    flexDirection: 'row',
    height: 65,
    backgroundColor: '#000',
    borderTopWidth: 1,
    borderTopColor: '#333',
    paddingBottom: 10,
  },
  tabItem: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 5,
  },
  activeTab: {
    borderTopWidth: 3,
    borderTopColor: '#00cc66',
  },
  tabText: {
    color: '#888',
    fontSize: 16,
    fontWeight: '600',
  },
  activeTabText: {
    color: '#00cc66',
  },
  floatingPanel: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(20, 20, 20, 0.95)',
    padding: 15,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#333',
  },
  panelTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  panelText: {
    color: '#bbb',
    fontSize: 14,
    marginBottom: 4,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  overlayText: {
    color: '#00cc66',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 30,
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 10,
    borderRadius: 8,
  },
  scanBox: {
    width: width * 0.65,
    height: width * 0.65,
    borderWidth: 3,
    borderColor: '#00cc66',
    backgroundColor: 'transparent',
    borderRadius: 10,
  }
});
